# README - Sequential Denoising + LLIE Evaluation Pipeline

**Master's Thesis Repository** — Modular evaluation framework for denoising and low-light image enhancement (LLIE) combinations, with joint ZS-N2N + COLIE and N2D + COLIE pipelines for endoscopic image restoration.

---

##  Overview

This repository contains the code for my Master's thesis evaluating sequential denoising and low-light image enhancement methods for endoscopic images (Endovis2018). It provides:

### 1. **Sequential Evaluation Pipeline**
Processes images through denoiser → LLIE combinations with automatic PSNR/SSIM evaluation:
- **5 Denoisers**: BM3D, ZS-N2N, N2D, N2V, N2Self
- **3 LLIE Methods**: COLIE, SCI, RUAS
- Any combination: Denoise-only, LLIE-only, or sequential

### 2. **Joint ZS-N2N + COLIE Pipeline**
Two-phase pipeline for endoscopic images:
- **Phase 1 — Bootstrap**: ZS-N2N denoiser trained from scratch on each noisy image
- **Phase 2 — Joint optimization**: COLIE illumination network trained alongside denoiser with feedback loss

### 3. **Joint N2D + COLIE Pipeline**
Three-phase pipeline with Noise2Detail:
- **Phase 1 — Bootstrap**: N2D denoiser trained from scratch on each noisy image
- **Phase 2 — Pixel shuffling**: Pseudo-ensemble with weighting
- **Phase 3 — Joint optimization**: COLIE trained alongside denoiser with self-supervised stage 2 loss

---

##  Installation

### Core Dependencies
```bash
pip install torch torchvision numpy pandas scikit-image tqdm pytorch-msssim opencv-python
```

### External Methods for the Sequential Evaluation
This part of the repository is a **wrapper** that calls existing implementations. You need to install the following separately:

| Method | Installation |
|--------|--------------|
| **BM3D** | `pip install bm3d` |
| **N2V** | `pip install n2v tensorflow` |
| **ZS-N2N, N2D, N2Self** | Install from their respective repositories |
| **COLIE, SCI, RUAS** | Clone repositories and ensure scripts are accessible |

Paths to external scripts can be set via command-line arguments (`--colie-path`, `--sci-path`, `--ruas-path`).

---

##  Usage

### Sequential Evaluation Pipeline

**Denoise + Enhance (with GT):**
```bash
python pipeline/seq_evaluation.py \
    --input-folder ./data/noisy_images \
    --gt-folder ./data/ground_truth \
    --denoiser N2D \
    --llie COLIE \
    --output-folder ./results/seq_eval \
```

Omit `--denoiser` for LLIE-only, omit `--llie` for denoise-only, or omit `--gt-folder` for processing without evaluation. Add `--save-images` to save output images.

### Joint ZS-N2N + COLIE Pipeline

```bash
python pipeline/zsn2n_colie_pipeline.py \
    --input-folder ./data/noisy_images \
    --gt-folder ./data/ground_truth \
    --output-folder ./results/zsn2n_colie \
```

### Joint N2D + COLIE Pipeline

```bash
python pipeline/n2d_colie_pipeline.py \
    --input-folder ./data/noisy_images \
    --gt-folder ./data/ground_truth \
    --output-folder ./results/n2d_colie \
```
Omit `--gt-folder` for processing without evaluation. Add `--save-images` to save output images.
Run `--help` with any script for the full list of arguments.

---

## Credits & References

This repository builds on the following published work. If you use this code, please also cite the original papers below.

**ZS-N2N**
> Mansour, Y. & Heckel, R. (2023). "Zero-Shot Noise2Noise: Efficient Image Denoising without any Data." CVPR 2023.
> [arXiv:2303.11253](https://arxiv.org/abs/2303.11253)

**COLIE**
> Chobola, T., Liu, Y., Zhang, H., Schnabel, J. A., & Peng, T. (2024). "Fast Context-Based Low-Light Image Enhancement via Neural Implicit Representations." ECCV 2024.
> [Paper](https://arxiv.org/abs/2407.12511) · [Code](https://github.com/ctom2/colie) (Apache 2.0)

**Noise2Detail (N2D)**
> Chobola, T. & Schnabel, J. A. (2025). "Lightweight Data-Free Denoising for Detail-Preserving Biomedical Image Restoration." MICCAI 2025.
> [Code](https://github.com/ctom2/noise2detail) (Apache 2.0)

**Noise2Void (N2V)**
> Krull, A., Buchholz, T.-O., & Jug, F. (2019). "Noise2Void — Learning Denoising from Single Noisy Images." CVPR 2019.
> [arXiv:1811.10980](https://arxiv.org/abs/1811.10980)

**Noise2Self (N2Self)**
> Batson, J. & Royer, L. (2019). "Noise2Self: Blind Denoising by Self-Supervision." ICML 2019.

Note: The original N2Self implementation was designed for grayscale images. This repository adapts it for RGB images using SharedMaskRGBNoise2Self.

**BM3D**
> Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering." IEEE Transactions on Image Processing.

**SCI**
> Ma, L., Ma, T., Liu, R., Fan, X., & Luo, Z. (2022). "Toward Fast, Flexible, and Robust Low-Light Image Enhancement." CVPR 2022.
> [arXiv:2204.10137](https://arxiv.org/abs/2204.10137)

**RUAS**
> Liu, R., Ma, L., Zhang, J., Fan, X., & Luo, Z. (2021). "Retinex-Inspired Unrolling with Cooperative Prior Architecture Search for Low-Light Image Enhancement." CVPR 2021.

---
### Third-Party Code
Portions of this repository adapt code from third-party sources, which remain subject to their original licenses:
- **COLIE** (`github.com/ctom2/colie`): Licensed under Apache License 2.0. Attribution and "state changes" requirements apply — see `THIRD_PARTY_LICENSES/colie-LICENSE` and inline notices in the code files.
- **Noise2Detail** (`github.com/ctom2/noise2detail`): Used with permission of the author.
- **ZS-N2N**: Adapted from reference implementation (Colab notebook) by Mansour & Heckel; no explicit open-source license.
- **Other methods** (BM3D, N2V, N2Self, SCI, RUAS): Called via their own installed packages or external scripts and remain subject to their respective upstream licenses.
---

## Acknowledgements

This work was conducted at the Helmholtz Institute Munich and the Technical University of Munich.

- **Supervisor**: Prof. Dr. Julia A. Schnabel
- **PhD Supervisor**: Tomáš Chobola
- **Research Group**: Dr. Tingying Peng's Group

Special thanks to Tomáš Chobola for his guidance during the thesis.

---

## Thesis Information

This repository is part of:
- **Degree**: Master's Thesis
- **Institution**: Technical University of Munich
- **Year**: 2026
- **Supervisor**: Prof. Dr. Julia A. Schnabel

---

**© 2026 [Nida Fareghzadeh]**

---

## 📁 Repository Structure

*See [STRUCTURE.md](STRUCTURE.md) for detailed directory layout.*
