"""
LLIE wrappers for sequential evaluation.
Calls external scripts (COLIE, SCI, RUAS).
"""

import subprocess
import tempfile
import shutil
from pathlib import Path


def run_colie(image_path, output_path, colie_script_path, alpha=1, beta=20, gamma=9, delta=5, L=0.1):
    """Run COLIE enhancement via external script."""
    colie_script = Path(colie_script_path).resolve()
    if not colie_script.exists():
        raise FileNotFoundError(f"COLIE script not found at {colie_script}")
    
    temp_dir = tempfile.mkdtemp()
    temp_input = Path(temp_dir) / "input"
    temp_output = Path(temp_dir) / "output"
    temp_input.mkdir(parents=True, exist_ok=True)
    temp_output.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(image_path, temp_input / image_path.name)
    
    cmd = [
        "python", str(colie_script),
        "--input_folder", str(temp_input.resolve()),
        "--output_folder", str(temp_output.resolve()),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--delta", str(delta),
        "--L", str(L)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(colie_script.parent))
        out_files = list(temp_output.glob("*.png"))
        if out_files:
            shutil.copy2(out_files[0], output_path)
        else:
            raise RuntimeError(f"No output found from COLIE for {image_path.name}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path


def run_sci(image_path, output_path, sci_script_path, weight='difficult'):
    """Run SCI enhancement via external script."""
    sci_script = Path(sci_script_path).resolve()
    if not sci_script.exists():
        raise FileNotFoundError(f"SCI script not found at {sci_script}")
    
    weights_path = sci_script.parent / "weights" / f"{weight}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"SCI weights not found at {weights_path}")
    
    temp_dir = tempfile.mkdtemp()
    temp_input = Path(temp_dir) / "input"
    temp_output = Path(temp_dir) / "output"
    temp_input.mkdir(parents=True, exist_ok=True)
    temp_output.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(image_path, temp_input / image_path.name)
    
    cmd = [
        "python", str(sci_script),
        "--data_path", str(temp_input.resolve()),
        "--save_path", str(temp_output.resolve()),
        "--model", str(weights_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(sci_script.parent))
        out_files = list(temp_output.glob("*.png"))
        if out_files:
            shutil.copy2(out_files[0], output_path)
        else:
            raise RuntimeError(f"No output found from SCI for {image_path.name}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path


def run_ruas(image_path, output_path, ruas_script_path, weight='upe'):
    """Run RUAS enhancement via external script."""
    ruas_script = Path(ruas_script_path).resolve()
    if not ruas_script.exists():
        raise FileNotFoundError(f"RUAS script not found at {ruas_script}")
    
    temp_dir = tempfile.mkdtemp()
    temp_input = Path(temp_dir) / "input"
    temp_output = Path(temp_dir) / "output"
    temp_input.mkdir(parents=True, exist_ok=True)
    temp_output.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(image_path, temp_input / image_path.name)
    
    cmd = [
        "python", str(ruas_script),
        "--data_path", str(temp_input.resolve()),
        "--save_path", str(temp_output.resolve()),
        "--model", str(weight)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(ruas_script.parent))
        out_files = list(temp_output.glob("*.png"))
        if out_files:
            shutil.copy2(out_files[0], output_path)
        else:
            raise RuntimeError(f"No output found from RUAS for {image_path.name}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_path