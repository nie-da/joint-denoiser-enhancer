import subprocess
import sys

cmd = [
    "python",
    "sequential_evaluation/seq_eval.py",
    "--input-folder", "./data/noisy_images/synthetic_dataset_V/G3/noise_50",
    "--gt-folder", "./data/gt",
    "--denoiser", "N2V",
    "--llie", "COLIE",
    "--max-images", "1",
    "--save-images",
    "--output-folder", "./test_output"
]

subprocess.run(cmd)