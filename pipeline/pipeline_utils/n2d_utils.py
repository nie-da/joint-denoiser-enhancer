# =============================================================================
# N2D model and losses
# Adapted from Chobola & Schnabel, MICCAI 2025.
# Code: https://github.com/ctom2/noise2detail  (Apache License 2.0)
#Adaptations: stage2_loss modified (self-supervised input)
# =============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




class DenoisingNetwork(nn.Module):
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


def pair_downsampler(img):
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device).repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device).repeat(c, 1, 1, 1)
    return F.conv2d(img, filter1, stride=2, groups=c), F.conv2d(img, filter2, stride=2, groups=c)


def stage1_loss(noisy_img, model):
    """ZS-N2N-style loss. model predicts a RESIDUAL."""
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
    return loss_res + loss_cons


def stage2_loss(semi_image, model):
    """N2D stage-2 loss, self-contained like stage1_loss.
    model predicts the DENOISED IMAGE DIRECTLY (no residual subtraction).
    semi_image acts as both input and self-supervised target."""
    semi1, semi2 = pair_downsampler(semi_image)
    pred1 = model(semi1)
    pred2 = model(semi2)
    loss_res = 0.5 * (F.mse_loss(semi1, pred2) + F.mse_loss(semi2, pred1))
    denoised = model(semi_image)
    denoised1, denoised2 = pair_downsampler(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
    return loss_res + loss_cons


def pixel_unshuffle(input, factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // factor
    out_width  = in_width  // factor
    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor, out_width, factor)
    batch_size *= factor ** 2
    return input_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(
        batch_size, channels, out_height, out_width)


def pixel_shuffle(input, factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height * factor
    out_width  = in_width  * factor
    batch_size //= factor ** 2
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height, in_width)
    return input_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(
        batch_size, channels, out_height, out_width)


def build_pseudo_ensemble(model1, img):
    """Multi-scale pixel-shuffle ensemble. model1 must already be frozen/eval."""
    with torch.no_grad():
        #direct = torch.clamp(img - model1(img), 0, 1)
        direct =img - model1(img)

        inp2   = pixel_unshuffle(img, 2)
        p2     = torch.clamp(inp2 - model1(inp2), 0, 1)
        p2     = pixel_shuffle(p2, 2)
        p2     = torch.clamp(p2 - model1(p2), 0, 1)
      
        inp4   = pixel_unshuffle(img, 4)
        p4     = torch.clamp(inp4 - model1(inp4), 0, 1)
        p4     = pixel_shuffle(p4, 4)
        p4     = torch.clamp(p4 - model1(p4), 0, 1)
    
    return (2*direct + p2 + p4) / 4

