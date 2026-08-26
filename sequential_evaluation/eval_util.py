"""
Evaluation utilities for sequential evaluation.
"""

import numpy as np
import skimage.io as io
from skimage.color import gray2rgb
import torch
import gc


def load_image(image_path):
    """Load image as numpy array in [0,1] range."""
    img = io.imread(image_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    return img


def save_image(img, save_path):
    """Save numpy image as PNG."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    io.imsave(save_path, img_uint8)


def evaluate_results(clean, processed):
    """Compute PSNR/SSIM between clean and processed images."""
    from pytorch_msssim import ssim
    
    clean_t = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0).float()
    processed_t = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).float()
    
    mse = torch.nn.functional.mse_loss(clean_t, processed_t).item()
    psnr = 10 * np.log10(1 / mse)
    ssim_val = ssim(clean_t, processed_t, data_range=1.0).item()
    return psnr, ssim_val


def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()