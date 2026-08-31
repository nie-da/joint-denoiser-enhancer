"""
N2D + COLIE joint denoising/enhancement pipeline.

Three-stage pipeline:
  Stage 1 — Bootstrap:  train model1 on the noisy image using a ZS-N2N-style
             residual loss.
  Stage 2 — Ensemble:   build a pseudo-clean label via a multi-scale
             pixel-shuffle ensemble from the frozen model1.
  Stage 3 — Joint opt:  jointly optimize a COLIE illumination-enhancement
             network and a second direct-prediction denoiser (model2) using
             a feedback loss between the two.

Ground truth is optional. When --gt-folder is provided, PSNR/SSIM are
computed and saved to CSV incrementally with resume support. Without GT the
pipeline runs and saves output images only (if --save-images is set).

Credits
-------
Pipeline design, CLI, and evaluation/IO code are original to this project.

  - DenoisingNetwork, pair_downsampler, stage1_loss, pixel_unshuffle,
    pixel_shuffle, build_pseudo_ensemble: adapted from N2D
    (Chobola & Schnabel, MICCAI 2025).
    Code: https://github.com/ctom2/noise2detail  (Apache License 2.0)
  - INF, L_exp, L_TV, rgb2hsv_torch, hsv2rgb_torch, and COLIE utilities
    (get_v_component, interpolate_image, get_coords, get_patches, filter_up,
    replace_v_component): adapted from COLIE (Chobola et al., ECCV 2024).
    Code: https://github.com/ctom2/colie  (Apache License 2.0)

See THIRD_PARTY_LICENSES/colie-LICENSE for the full Apache 2.0 license text.

Usage:
    # With ground truth (saves CSV only)
    python n2d_colie_pipeline.py \
        --input-folder ./noisy --gt-folder ./clean --output-folder ./out

    # With ground truth and save images
    python n2d_colie_pipeline.py \
        --input-folder ./noisy --gt-folder ./clean --output-folder ./out --save-images

    # Without ground truth (save images only)
    python n2d_colie_pipeline.py \
        --input-folder ./noisy --output-folder ./out --save-images

    # Without ground truth, no saving (just process)
    python n2d_colie_pipeline.py \
        --input-folder ./noisy --output-folder ./out
"""

import sys
import argparse
import random
from pathlib import Path
import os

import numpy as np
import pandas as pd
import skimage.io as io
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.color import gray2rgb
from tqdm import tqdm

from pipeline_utils.common_utils import load_image, save_image, evaluate_results, DEVICE
from pipeline_utils.n2d_utils import DenoisingNetwork, stage1_loss, stage2_loss, build_pseudo_ensemble

# ============================================================
# COLIE IMPORTS
# ============================================================
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'colie'))
from color import rgb2hsv_torch, hsv2rgb_torch
from loss import L_exp, L_TV
from siren import INF
from utils import get_v_component, interpolate_image, get_coords, get_patches, filter_up, replace_v_component


# =============================================================================
# Argument parser
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N2D + COLIE joint denoising/enhancement pipeline."
    )

    # Paths
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing input PNG images.")
    parser.add_argument("--gt-folder", type=str, default=None,
                        help="Optional folder containing ground-truth images "
                             "(filenames must match input images exactly). "
                             "When omitted, evaluation is skipped and no CSV "
                             "is written.")
    parser.add_argument("--output-folder", type=str, default="./output_n2d_colie",
                        help="Folder to write output images and (if GT provided) "
                             "results CSV. Default: ./output_n2d_colie")

    # Training schedule
    parser.add_argument("--bootstrap-epochs", type=int, default=200,
                        help="Stage 1: model1 bootstrap epochs. Default: 200")
    parser.add_argument("--joint-epochs", type=int, default=200,
                        help="Stage 3: joint COLIE + model2 epochs. Default: 200")
    parser.add_argument("--lambda-n2n", type=float, default=0.5,
                        help="Weight of the Stage 2 denoiser loss in the joint "
                             "objective. Default: 0.5")
    parser.add_argument("--lambda-feedback", type=float, default=30.0,
                        help="Weight of the denoiser-to-COLIE feedback loss. "
                             "Default: 30.0")

    # Optimizer (Stage 1)
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate for the Stage 1 optimizer. "
                             "Default: 0.0005")
    parser.add_argument("--step-size", type=int, default=1000,
                        help="StepLR step size for the Stage 1 scheduler. "
                             "Default: 1000")
    parser.add_argument("--lr-gamma", type=float, default=0.5,
                        help="StepLR decay factor for the Stage 1 scheduler. "
                             "Default: 0.5")

    # COLIE loss weights
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Spatial loss weight. Default: 1.0")
    parser.add_argument("--beta", type=float, default=20.0,
                        help="TV loss weight. Default: 20.0")
    parser.add_argument("--gamma-colie", type=float, default=9.0,
                        help="Exposure loss weight. Default: 9.0")
    parser.add_argument("--delta", type=float, default=5.0,
                        help="Sparsity loss weight. Default: 5.0")
    parser.add_argument("--exposure-level", type=float, default=0.1,
                        help="Target exposure level L for the exposure loss. "
                             "Default: 0.1")

    # Model / patch settings
    parser.add_argument("--down-size", type=int, default=256,
                        help="Side length for V-channel downsampling in COLIE. "
                             "Default: 256")
    parser.add_argument("--window", type=int, default=7,
                        help="Patch window size for COLIE input patches. "
                             "Default: 7")
    parser.add_argument("--chan-embed", type=int, default=48,
                        help="Embedding channels in the denoising networks. "
                             "Default: 48")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap on number of images to process (for testing).")
    parser.add_argument("--save-images", action="store_true",
                        help="Save output images to disk. Default: False "
                             "(only CSV is saved if GT provided).")

    return parser


# =============================================================================
# Process single image
# =============================================================================

def process_image(img_path: Path, args: argparse.Namespace):
    """Run the full three-stage pipeline on a single image.

    Returns
    -------
    final_output : torch.Tensor  shape (1, C, H, W), float32, [0, 1]
    noisy_img    : torch.Tensor  shape (1, C, H, W), float32, [0, 1]
    """
    noisy_np = load_image(img_path)
    noisy_img = noisy_np.to(DEVICE)

    # ── Stage 1: bootstrap ──
    model1 = DenoisingNetwork(noisy_img.shape[1], chan_embed=args.chan_embed).to(DEVICE)
    optimizer_s1 = optim.Adam(model1.parameters(), lr=args.lr)
    scheduler_s1 = optim.lr_scheduler.StepLR(
        optimizer_s1, step_size=args.step_size, gamma=args.lr_gamma)

    for _ in range(args.bootstrap_epochs):
        optimizer_s1.zero_grad()
        stage1_loss(noisy_img, model1).backward()
        optimizer_s1.step()
        scheduler_s1.step()

    # ── Stage 2: pixel-shuffle ensemble ──
    for p in model1.parameters():
        p.requires_grad_(False)
    model1.eval()

    semi_bootstrap = build_pseudo_ensemble(model1, noisy_img)

    # Precompute COLIE inputs from the pseudo-clean ensemble output
    with torch.no_grad():
        colie_hsv = rgb2hsv_torch(semi_bootstrap)
        colie_v = get_v_component(colie_hsv)
        colie_v_lr = interpolate_image(colie_v, args.down_size, args.down_size)
        colie_coords = get_coords(args.down_size, args.down_size)
        colie_patches = get_patches(colie_v_lr, args.window)

    # ── Stage 3: joint COLIE <-> model2 ──
    model2 = DenoisingNetwork(noisy_img.shape[1], chan_embed=args.chan_embed).to(DEVICE)
    model_enhancer = INF(
        patch_dim=args.window ** 2,
        num_layers=4,
        hidden_dim=256,
        add_layer=2,
    ).to(DEVICE)

    optimizer_n2d = optim.Adam(model2.parameters(), lr=1e-3)
    optimizer_colie = optim.Adam(model_enhancer.parameters(), lr=1e-5)

    l_exp = L_exp(16, args.exposure_level)
    l_TV = L_TV()

    for _ in range(args.joint_epochs):

        # Step 1: COLIE illumination estimation
        illu_res_lr = model_enhancer(colie_patches, colie_coords).view(
            1, 1, args.down_size, args.down_size)
        illu_lr = illu_res_lr + colie_v_lr
        img_v_fixed_lr = torch.clamp(colie_v_lr / illu_lr.clamp(min=1e-4), 0, 1)

        loss_spa = torch.mean(torch.abs(torch.pow(illu_lr - colie_v_lr, 2)))
        loss_tv = l_TV(illu_lr)
        loss_exp = torch.mean(l_exp(illu_lr))
        loss_sparsity = torch.mean(img_v_fixed_lr)
        loss_colie = (
            loss_spa * args.alpha +
            loss_tv * args.beta +
            loss_exp * args.gamma_colie +
            loss_sparsity * args.delta
        )

        # Step 2: V transform + HSV -> RGB
        with torch.no_grad():
            img_v_fixed = filter_up(colie_v_lr, img_v_fixed_lr, colie_v).clamp(0, 1)
            img_hsv_fixed = replace_v_component(colie_hsv.clone(), img_v_fixed)
            enhanced = torch.clamp(hsv2rgb_torch(img_hsv_fixed), 0, 1)

        # Step 3: N2D Stage 2 loss on the enhanced image
        # model2 is a direct predictor (no residual subtraction)
        loss_n2n = stage2_loss(enhanced, model2)

        # Step 4: model2 -> COLIE feedback
        with torch.no_grad():
            denoised_out = torch.clamp(model2(enhanced), 0, 1)
            denoised_hsv = rgb2hsv_torch(denoised_out)
            denoised_v = get_v_component(denoised_hsv)
            denoised_v_lr = interpolate_image(denoised_v, args.down_size, args.down_size)

        loss_feedback = F.mse_loss(
            illu_lr / illu_lr.mean().clamp(min=1e-4),
            denoised_v_lr.detach() / denoised_v_lr.mean().clamp(min=1e-4),
        )

        # Step 5: combined backward
        optimizer_colie.zero_grad()
        optimizer_n2d.zero_grad()
        loss_total = (
            loss_colie
            + args.lambda_n2n * loss_n2n
            + args.lambda_feedback * loss_feedback
        )
        loss_total.backward()
        optimizer_colie.step()
        optimizer_n2d.step()

    # ── Final output ──
    with torch.no_grad():
        illu_res = model_enhancer(colie_patches, colie_coords).view(
            1, 1, args.down_size, args.down_size)
        illu_lr = illu_res + colie_v_lr
        img_v_fixed_lr = torch.clamp(colie_v_lr / illu_lr.clamp(min=1e-4), 0, 1)
        img_v_fixed = filter_up(colie_v_lr, img_v_fixed_lr, colie_v).clamp(0, 1)
        img_hsv_fixed = replace_v_component(colie_hsv.clone(), img_v_fixed)
        enhanced_final = torch.clamp(hsv2rgb_torch(img_hsv_fixed), 0, 1)
        final_output = torch.clamp(model2(enhanced_final), 0, 1)

    return final_output, noisy_img


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    gt_folder = Path(args.gt_folder) if args.gt_folder else None

    output_folder.mkdir(parents=True, exist_ok=True)

    img_files = sorted(input_folder.glob("*.png"))
    if args.max_images:
        img_files = img_files[:args.max_images]

    # Resume support (only relevant when GT is provided and CSV exists)
    results_list = []
    csv_path = None
    completed = set()

    if gt_folder is not None:
        csv_path = output_folder / "results.csv"
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            completed = set(df_existing["filename"].tolist())
            results_list = df_existing.to_dict("records")
            print(f"Resuming — {len(completed)} images already completed.")

    print(f"\n{'='*60}")
    print(f"N2D + COLIE pipeline")
    print(f"Input   : {input_folder}")
    print(f"GT      : {gt_folder if gt_folder else 'not provided (evaluation skipped)'}")
    print(f"Output  : {output_folder}")
    print(f"Images  : {len(img_files)}")
    print(f"Save images: {args.save_images}")
    print(f"{'='*60}\n")

    for img_path in tqdm(img_files, desc="Processing"):
        img_name = img_path.name

        if img_name in completed:
            print(f"  Skipping {img_name} (already completed)")
            continue

        try:
            final_output, noisy_img = process_image(img_path, args)

            # Save output image (optional)
            if args.save_images:
                output_np = (
                    final_output.squeeze(0).permute(1, 2, 0)
                    .cpu().detach().numpy() * 255
                ).astype(np.uint8)
                io.imsave(output_folder / img_name, output_np)

            # Evaluate only if GT folder was provided
            if gt_folder is not None:
                gt_path = gt_folder / img_name
                if gt_path.exists():
                    clean_np = io.imread(gt_path)
                    if clean_np.ndim == 2:
                        clean_np = gray2rgb(clean_np)
                    clean_img = (
                        torch.from_numpy(clean_np)
                        .permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                    )
                    psnr_n, psnr_d, ssim_n, ssim_d = evaluate_results(
                        clean_img, noisy_img, final_output)
                    results_list.append({
                        "filename": img_name,
                        "psnr_noisy": psnr_n,
                        "psnr_denoised": psnr_d,
                        "ssim_noisy": ssim_n,
                        "ssim_denoised": ssim_d,
                    })
                    pd.DataFrame(results_list).to_csv(csv_path, index=False)
                    print(f"  {img_name} — PSNR: {psnr_d:.2f} dB | SSIM: {ssim_d:.4f}")
                else:
                    print(f"  {img_name} — GT not found at {gt_path}, skipping evaluation")
            else:
                if args.save_images:
                    print(f"  {img_name} — saved")
                else:
                    print(f"  {img_name} — processed (no GT, no save)")

        except Exception as e:
            import traceback
            print(f"  Error processing {img_name}: {e}")
            traceback.print_exc()
            continue

    # Summary
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(csv_path, index=False)
        print("\n" + "=" * 60)
        print("SUMMARY — N2D + COLIE pipeline")
        print("=" * 60)
        print(f"Images processed : {len(df)}")
        print(f"Mean PSNR noisy  : {df['psnr_noisy'].mean():.2f} dB")
        print(f"Mean PSNR output : {df['psnr_denoised'].mean():.2f} dB")
        print(f"Mean SSIM noisy  : {df['ssim_noisy'].mean():.4f}")
        print(f"Mean SSIM output : {df['ssim_denoised'].mean():.4f}")
        print(f"Results saved to : {csv_path}")
    else:
        print(f"\nDone. {len(img_files)} images processed (GT not provided, no evaluation).")


if __name__ == "__main__":
    main()