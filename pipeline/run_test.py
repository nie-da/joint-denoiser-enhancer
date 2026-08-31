import subprocess
import sys

cmd = [
    "python",
    "pipeline/n2d_pipeline.py",
    "--input-folder", "./data/noisy_images/synthetic_dataset_V/G3/noise_50",
    "--output-folder", "./out",
    "--save-images",

]

subprocess.run(cmd)