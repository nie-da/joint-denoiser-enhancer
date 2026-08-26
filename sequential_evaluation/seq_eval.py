"""
Sequential evaluation script for denoising + LLIE combinations.

Usage:
    python seq_evaluation.py --input-folder ./images --gt-folder ./gt \
        --denoiser N2D --llie COLIE

All denoisers work blindly — no noise level is required.
"""

import argparse
import tempfile
import shutil
import traceback
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from denoise_wrapper import get_denoiser
from llie_wrapper import run_colie, run_sci, run_ruas
from eval_util import load_image, save_image, evaluate_results, cleanup_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential denoising + LLIE evaluation.")
    parser.add_argument("--input-folder", type=str, required=True,
                        help="Folder containing input images.")
    parser.add_argument("--gt-folder", type=str, default=None,
                        help="Optional folder containing ground truth images.")
    parser.add_argument("--denoiser", type=str, default=None,
                        choices=['BM3D', 'ZS-N2N', 'N2D', 'N2V', 'N2Self'],
                        help="Denoising method to apply.")
    parser.add_argument("--llie", type=str, default=None,
                        choices=['COLIE', 'SCI', 'RUAS'],
                        help="LLIE method to apply.")
    parser.add_argument("--llie-weight", type=str, default=None,
                        help="Weight for SCI (easy/medium/difficult) or RUAS (upe/dark/lol).")
    parser.add_argument("--output-folder", type=str, default="./results",
                        help="Output folder for results.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save processed images.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process (for testing).")
    parser.add_argument("--denoiser-epochs", type=int, default=2000,
                        help="Epochs for denoiser training (ZS-N2N, N2D).")
    
    # External tool paths
    parser.add_argument("--colie-path", type=str, default="external/colie/colie.py",
                        help="Path to COLIE script.")
    parser.add_argument("--sci-path", type=str, default="external/SCI/CVPR/test.py",
                        help="Path to SCI script.")
    parser.add_argument("--ruas-path", type=str, default="external/RUAS/test.py",
                        help="Path to RUAS script.")
    
    return parser.parse_args()


def run():
    args = parse_args()
    
    input_folder = Path(args.input_folder)
    gt_folder = Path(args.gt_folder) if args.gt_folder else None
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(input_folder.glob("*.png"))
    if args.max_images:
        img_files = img_files[:args.max_images]
    
    # Build method name
    method_parts = []
    if args.denoiser:
        method_parts.append(args.denoiser)
    if args.llie:
        method_parts.append(args.llie)
    method_name = "_".join(method_parts) if method_parts else "Noisy"
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(img_files)} images")
    print(f"Method: {method_name}")
    print(f"GT available: {gt_folder is not None}")
    print(f"{'='*60}\n")
    
    for img_path in tqdm(img_files, desc="Processing"):
        img_name = img_path.name
        
        try:
            temp_dir = tempfile.mkdtemp()
            temp_img_path = Path(temp_dir) / img_name
            
            img = load_image(img_path)
            
            # Apply denoiser if specified (blind — no noise level needed)
            if args.denoiser:
                denoiser_fn = get_denoiser(args.denoiser)
                
                # BM3D uses default sigma (0.1) — blind denoising
                if args.denoiser == 'BM3D':
                    processed = denoiser_fn(img, sigma=0.1)
                else:
                    processed = denoiser_fn(img, epochs=args.denoiser_epochs)
                
                save_image(processed, temp_img_path)
            else:
                save_image(img, temp_img_path)
            
            # Apply LLIE if specified
            if args.llie:
                temp_llie_path = Path(temp_dir) / f"llie_{img_name}"
                
                if args.llie == 'COLIE':
                    run_colie(temp_img_path, temp_llie_path, args.colie_path)
                elif args.llie == 'SCI':
                    weight = args.llie_weight if args.llie_weight else 'difficult'
                    run_sci(temp_img_path, temp_llie_path, args.sci_path, weight=weight)
                elif args.llie == 'RUAS':
                    weight = args.llie_weight if args.llie_weight else 'upe'
                    run_ruas(temp_img_path, temp_llie_path, args.ruas_path, weight=weight)
                
                final_img = load_image(temp_llie_path)
            else:
                final_img = load_image(temp_img_path)
            
            # Save final output
            if args.save_images:
                output_path = output_folder / f"{method_name}_{img_name}"
                save_image(final_img, output_path)
            
            # Evaluate if GT available
            if gt_folder is not None:
                gt_path = gt_folder / img_name
                if gt_path.exists():
                    gt = load_image(gt_path)
                    psnr, ssim = evaluate_results(gt, final_img)
                    results.append({
                        'filename': img_name,
                        'method': method_name,
                        'psnr': psnr,
                        'ssim': ssim,
                    })
                    print(f"  {img_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
                else:
                    print(f"  ⚠️ GT not found for {img_name}")
                    results.append({
                        'filename': img_name,
                        'method': method_name,
                        'psnr': None,
                        'ssim': None,
                    })
            else:
                print(f"  ✅ {img_name} processed (no GT)")
                results.append({
                    'filename': img_name,
                    'method': method_name,
                    'psnr': None,
                    'ssim': None,
                })
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            cleanup_gpu()
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            traceback.print_exc()
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            cleanup_gpu()
            continue
    
    # Save results
    csv_filename = f"results_{method_name}.csv"
    csv_path = output_folder / csv_filename
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY — {method_name}")
    print("="*60)
    print(f"Images processed: {len(results)}/{len(img_files)}")
    
    if gt_folder is not None and 'psnr' in df.columns:
        valid_results = df[df['psnr'].notna()]
        if len(valid_results) > 0:
            print(f"Mean PSNR: {valid_results['psnr'].mean():.2f} dB")
            print(f"Mean SSIM: {valid_results['ssim'].mean():.4f}")
    
    print(f"Results saved to: {csv_path}")
    
    return df


if __name__ == "__main__":
    run()