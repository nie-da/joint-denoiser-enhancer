import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Noise functions ---
def add_noise(x, noise_type='gauss', noise_level=25):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level/255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level
    return noisy

# --- Model definition ---
class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

# --- Downsampler ---
# --- Downsampler ---
def pair_downsampler(img):
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0,0.5],[0.5,0]]]]).to(img.device).repeat(c,1,1,1)
    filter2 = torch.FloatTensor([[[[0.5,0],[0,0.5]]]]).to(img.device).repeat(c,1,1,1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2  # <-- no visualization call here

# --- Visualization of downsampled images ---
def show_downsampled(noisy_img):
    img1, img2 = pair_downsampler(noisy_img)
    
    # Convert tensors to HWC for display
    def to_display(x):
        x = x.cpu().squeeze(0)
        if x.ndim == 3:  # C H W
            x = x.permute(1, 2, 0)
        return x.numpy()
    
    img0 = to_display(noisy_img)
    img1 = to_display(img1)
    img2 = to_display(img2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    # Decide colormap based on channels
    cmap0 = None if img0.shape[-1] > 1 else 'gray'
    cmap1 = None if img1.shape[-1] > 1 else 'gray'
    cmap2 = None if img2.shape[-1] > 1 else 'gray'

    ax[0].imshow(img0, cmap=cmap0)
    ax[0].set_title('Noisy Img')
    ax[1].imshow(img1, cmap=cmap1)
    ax[1].set_title('First downsampled')
    ax[2].imshow(img2, cmap=cmap2)
    ax[2].set_title('Second downsampled')
    plt.show()


# --- Loss and training functions ---
def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)

def loss_func(noisy_img, model):
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
    return loss_res + loss_cons

def train_step(model, optimizer, noisy_img):
    loss = loss_func(noisy_img, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        mse_val = mse(clean_img, pred).item()
        psnr = 10*np.log10(1/mse_val)
    return psnr

def denoise(model, noisy_img):
    with torch.no_grad():
        return torch.clamp(noisy_img - model(noisy_img), 0, 1)

# --- Utility to load PNG as tensor ---
def load_image_as_tensor(img_path, device):
    img = Image.open(img_path)
    if img.mode == 'L':
        img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0).float()/255.0
    else:
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float()/255.0

    img_tensor = img_tensor.to(device)
    print(f"Loaded {img_path.name}, shape: {img_tensor.shape}, device: {device}")
    return img_tensor


def show_images_with_axis(clean_img, noisy_img, denoised_img, noisy_psnr=None, denoised_psnr=None):
    def to_display(img_tensor):
        img_np = img_tensor.cpu().squeeze(0).numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1,3]:
            img_np = np.transpose(img_np, (1,2,0))
        if img_np.shape[-1] == 1:
            img_np = img_np[:,:,0]
        return img_np

    clean_np = to_display(clean_img)
    noisy_np = to_display(noisy_img)
    denoised_np = to_display(denoised_img)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))

    # Ground Truth
    ax[0].imshow(clean_np, cmap='gray' if clean_np.ndim==2 else None)
    ax[0].axis('on')
    ax[0].set_title("Ground Truth", fontsize=14)

    # Noisy Image
    ax[1].imshow(noisy_np, cmap='gray' if noisy_np.ndim==2 else None)
    ax[1].axis('on')
    ax[1].set_title("Noisy Image", fontsize=14)
    if noisy_psnr is not None:
        # Set x-axis ticks to show PSNR
        ax[1].set_xlabel(f"PSNR: {noisy_psnr:.2f} dB", fontsize=12)
        # Optionally force the label to be visible
        ax[1].xaxis.label.set_color('red')
        ax[1].xaxis.label.set_fontsize(12)

    # Denoised Image
    ax[2].imshow(denoised_np, cmap='gray' if denoised_np.ndim==2 else None)
    ax[2].axis('on')
    ax[2].set_title("Denoised Image", fontsize=14)
    if denoised_psnr is not None:
        ax[2].set_xlabel(f"PSNR: {denoised_psnr:.2f} dB", fontsize=12)
        ax[2].xaxis.label.set_color('green')

    plt.tight_layout()
    plt.show()


