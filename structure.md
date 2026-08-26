# Repository Structure
```
joint-denoiser-enhancer/
│
├── README.md
├── STRUCTURE.md
├── .gitignore
├── .gitmodules
│
├── THIRD_PARTY_LICENSES/
│   ├── colie-LICENSE
│   └── noise2detail-LICENSE
│
├── pipeline/
│   ├── __init__.py
│   ├── zsn2n_colie_pipeline.py
│   ├── n2d_colie_pipeline.py
│   ├── common_utils.py
│   ├── zsn2n_utils.py
│   └── n2d_utils.py
│
├── external/
│   ├── colie/
│   ├── noise2detail/
│   ├── N2V/
│   ...
│   ├── SCI/
│   └── RUAS/
│
├── sequential_evaluation/
│   ├── seq_evaluation.py
│   └── aggregate_results.py
│
├── data/
│   ├── noisy_images/
│   ├── ground_truth/
│   ├── segmentation_eval/
│   └── segmentation_train/
│
 ── results/
    ├── seq_evaluation/
    ├── zsn2n_colie/
    └── n2d_colie/
```

