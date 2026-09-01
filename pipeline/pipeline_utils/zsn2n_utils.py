# =============================================================================
# ZS-N2N model and losses from Mansour & Heckel
# Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b
# ======================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F



# -----------------------------
# ZS-N2N model
# -----------------------------
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

def mse(gt, pred):
    return nn.MSELoss()(gt, pred)

def pair_downsampler(img):
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device).repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device).repeat(c, 1, 1, 1)
    return F.conv2d(img, filter1, stride=2, groups=c), F.conv2d(img, filter2, stride=2, groups=c)

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

def denoise(model, img):
    with torch.no_grad():
        return torch.clamp(img - model(img), 0, 1)
