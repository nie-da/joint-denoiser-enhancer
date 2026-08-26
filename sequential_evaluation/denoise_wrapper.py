"""
Denoiser wrappers for sequential evaluation.
Imports and returns denoiser functions.
"""

import torch
import gc
import numpy as np
import skimage.io as io
from pathlib import Path
import tqdm
import sys

# If tqdm.notebook is used, redirect it to standard tqdm
class TqdmPatched:
    def __getattr__(self, name):
        return getattr(tqdm, name)

sys.modules['tqdm.notebook'] = TqdmPatched()


def get_denoiser(name):
    """
    Import and return denoiser function based on name.
    Denoisers must be installed separately.
    """
    if name == 'BM3D':
        try:
            from bm3d import bm3d
            def run_bm3d(image, sigma=0.1):
                if image.ndim == 3:
                    denoised = np.zeros_like(image)
                    for c in range(image.shape[2]):
                        denoised[:, :, c] = bm3d(image[:, :, c], sigma_psd=sigma)
                else:
                    denoised = bm3d(image, sigma_psd=sigma)
                return np.clip(denoised, 0, 1)
            return run_bm3d
        except ImportError:
            raise ImportError("BM3D not installed. Please install: pip install bm3d")
    
    elif name == 'ZS-N2N':
        try:
            
            from zsn2n import Network, train_step, denoise
            
            def run_zsn2n(image, epochs=2000, lr=0.0005, step_size=1000, gamma=0.5, **kwargs):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                n_chan = img_tensor.shape[1]
                model = Network(n_chan).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                for _ in range(epochs):
                    loss = train_step(model, optimizer, img_tensor)
                    scheduler.step()
                denoised = denoise(model, img_tensor)
                del model, optimizer, scheduler
                torch.cuda.empty_cache()
                return denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return run_zsn2n
        except ImportError as e:
            raise ImportError(f"ZS-N2N not installed. Please ensure zsn2n.py is in sequential_evaluation/. Error: {e}")
        
    elif name == 'N2D':
        try:
            import os
            n2d_path = os.path.join(os.getcwd(), 'external/noise2detail')
            if n2d_path not in sys.path:
                sys.path.insert(0, n2d_path)
            
            # Import from the actual files
            from model import DenoisingNetwork
            from train import train_model, train_model_2
            from utils import pixel_unshuffle, pixel_shuffle

           
            def build_pseudo_ensemble(model, noisy_img):
                """
                Custom pseudo-ensemble for N2D.
                Weighting: (direct + pseudo1 + pseudo2) / 3
                """
                with torch.no_grad():
                    # Direct denoising
                    denoised_img_1 = torch.clamp(noisy_img - model(noisy_img), 0, 1)
                    
                    # Pseudo-ensemble at scale 2
                    input = pixel_unshuffle(noisy_img, 2)
                    pseudo1 = torch.clamp(input - model(input), 0, 1)
                    pseudo1 = pixel_shuffle(pseudo1, 2)
                    pseudo1 = torch.clamp(pseudo1 - model(pseudo1), 0, 1)
                    
                    # Pseudo-ensemble at scale 4
                    input = pixel_unshuffle(noisy_img, 4)
                    pseudo2 = torch.clamp(input - model(input), 0, 1)
                    pseudo2 = pixel_shuffle(pseudo2, 4)
                    pseudo2 = torch.clamp(pseudo2 - model(pseudo2), 0, 1)
                    
                    # Average
                    return (denoised_img_1 + pseudo1 + pseudo2) / 3
            
            def run_n2d(image, epochs=2000, lr=0.0005, step_size=1000, gamma=0.5, **kwargs):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                n_chan = img_tensor.shape[1]
                
                # Stage 1: Train first model (only 2 arguments: model, noisy_image)
                model1 = DenoisingNetwork(n_chan).to(device)
                train_model(model1, img_tensor, epochs=epochs)
                
                # Build pseudo-ensemble
                semi_bootstrap = build_pseudo_ensemble(model1, img_tensor)
                
                # Stage 2: Train second model (3 arguments: model, semi_image, noisy_image)
                model2 = DenoisingNetwork(n_chan).to(device)
                train_model_2(model2, semi_bootstrap, img_tensor, epochs=epochs // 2)
                
                with torch.no_grad():
                    denoised = torch.clamp(model2(semi_bootstrap), 0, 1)
                
                del model1, model2
                torch.cuda.empty_cache()
                return denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return run_n2d
        except ImportError as e:
            raise ImportError(f"N2D not installed. Please check external/noise2detail/ folder. Error: {e}")
    
    elif name == 'N2V':
        def run_n2v(image, epochs=250, **kwargs):
            import subprocess
            import tempfile
            import os
            
            # Use the N2V environment Python path
            tf_python = r'C:\Users\<USERNAME>\miniconda3\envs\n2v_env\python.exe'
            
            # Save input image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
                input_path = tmp_input.name
                io.imsave(input_path, (image * 255).astype(np.uint8))
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_output:
                output_path = tmp_output.name
            
            script_path = Path(__file__).parent / 'run_n2v_external.py'
            
            cmd = [
                tf_python,
                str(script_path),
                input_path,
                output_path,
                str(epochs)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                denoised = io.imread(output_path)
                if denoised.max() > 1:
                    denoised = denoised.astype(np.float32) / 255.0
                
                return np.clip(denoised, 0, 1)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"N2V failed: {e.stderr}")
            finally:
                try:
                    os.unlink(input_path)
                except:
                    pass
                try:
                    os.unlink(output_path)
                except:
                    pass
        
        return run_n2v
    
    elif name == 'N2Self':
        try:
            import sys
            import os
            
            # Add the external noise2self folder to path
            n2self_path = os.path.join(os.getcwd(), 'external', 'noise2self')
            if n2self_path not in sys.path:
                sys.path.insert(0, n2self_path)
            
            # Import from the external repository
            from mask import Masker
            from models.dncnn import DnCNN
            from torch.nn import MSELoss
            from torch.optim import Adam
            
            def run_n2self(image, epochs=500, mask_width=4, **kwargs):
                import torch.nn.functional as F
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)
                
                # Convert to tensor
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                B, C, H, W = img_tensor.shape
                
                # Setup
                torch.manual_seed(3)
                model = DnCNN(1, num_of_layers=8).to(device)
                optimizer = Adam(model.parameters(), lr=0.01)
                loss_fn = MSELoss()
                masker = Masker(width=mask_width, mode='interpolate')
                
                # Interpolation kernel for masked pixels
                kernel = torch.tensor([[0.5, 1.0, 0.5],
                                    [1.0, 0.0, 1.0],
                                    [0.5, 1.0, 0.5]], device=device).float()
                kernel = kernel.view(1, 1, 3, 3) / kernel.sum()
                
                best_val_loss = float('inf')
                best_denoised = None
                
                for i in range(epochs):
                    model.train()
                    
                    # Get shared mask for all channels
                    _, mask_single = masker.mask(img_tensor[:, 0:1, :, :], i % (masker.n_masks - 1))
                    mask_rgb = mask_single.repeat(1, 3, 1, 1)
                    mask_inv_rgb = 1 - mask_rgb
                    
                    # Apply mask with interpolation
                    if masker.mode == 'interpolate':
                        filtered_channels = []
                        for c in range(3):
                            ch = img_tensor[:, c:c+1, :, :]
                            filtered_c = F.conv2d(ch, kernel, stride=1, padding=1)
                            filtered_channels.append(filtered_c)
                        filtered_tensor = torch.cat(filtered_channels, dim=1)
                        masked_input = filtered_tensor * mask_rgb + img_tensor * mask_inv_rgb
                    else:
                        masked_input = img_tensor * mask_inv_rgb
                    
                    # Forward pass (reshape for single-channel model)
                    masked_input_flat = masked_input.view(B * C, 1, H, W)
                    output_flat = model(masked_input_flat)
                    output = output_flat.view(B, C, H, W)
                    
                    # Loss on masked pixels only
                    loss = loss_fn(output * mask_rgb, img_tensor * mask_rgb)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Validation every 10 epochs
                    if i % 10 == 0:
                        model.eval()
                        with torch.no_grad():
                            _, val_mask_single = masker.mask(img_tensor[:, 0:1, :, :], masker.n_masks - 1)
                            val_mask_rgb = val_mask_single.repeat(1, 3, 1, 1)
                            val_mask_inv_rgb = 1 - val_mask_rgb
                            
                            if masker.mode == 'interpolate':
                                filtered_channels_val = []
                                for c in range(3):
                                    ch = img_tensor[:, c:c+1, :, :]
                                    filtered_c_val = F.conv2d(ch, kernel, stride=1, padding=1)
                                    filtered_channels_val.append(filtered_c_val)
                                filtered_tensor_val = torch.cat(filtered_channels_val, dim=1)
                                masked_input_val = filtered_tensor_val * val_mask_rgb + img_tensor * val_mask_inv_rgb
                            else:
                                masked_input_val = img_tensor * val_mask_inv_rgb
                            
                            masked_input_val_flat = masked_input_val.view(B * C, 1, H, W)
                            output_val_flat = model(masked_input_val_flat)
                            output_val = output_val_flat.view(B, C, H, W)
                            val_loss = loss_fn(output_val * val_mask_rgb, img_tensor * val_mask_rgb)
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_denoised = model(img_tensor.view(B * C, 1, H, W)).view(B, C, H, W).detach().clone()
                
                if best_denoised is None:
                    best_denoised = model(img_tensor.view(B * C, 1, H, W)).view(B, C, H, W).detach()
                
                denoised = torch.clamp(best_denoised, 0, 1)
                
                del model, optimizer
                torch.cuda.empty_cache()
                
                return denoised.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            return run_n2self
        except Exception as e:
            raise ImportError(f"N2Self not installed. Make sure external/noise2self/ is cloned. Error: {e}")
        
    else:
        raise ValueError(f"Unknown denoiser: {name}. Available: BM3D, ZS-N2N, N2D, N2V, N2Self")