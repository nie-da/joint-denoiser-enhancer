import skimage.io as io
import numpy as np
from skimage.color import gray2rgb
import torch
from pytorch_msssim import ssim    
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def load_image(image_path):
    """Load image as tensor in [0,1] range."""
    
    img = io.imread(image_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    return img_tensor


def save_image(tensor, save_path):
    """Save tensor as PNG."""
    
    
    img_np = (tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    io.imsave(save_path, img_np)


def evaluate_results(clean, noisy, denoised):
    """Evaluate PSNR and SSIM for noisy and denoised images against clean image."""
    mse_noisy    = F.mse_loss(clean, noisy).item()
    mse_denoised = F.mse_loss(clean, denoised).item()
    psnr_noisy    = 10 * np.log10(1 / mse_noisy)
    psnr_denoised = 10 * np.log10(1 / mse_denoised)
    ssim_noisy    = ssim(clean, noisy,    data_range=1.0).item()
    ssim_denoised = ssim(clean, denoised, data_range=1.0).item()
    return psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised
