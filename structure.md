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
│
├── pipeline/
│   ├── zsn2n_pipeline.py
│   ├── n2d_pipeline.py
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
│   ├── seq_eval.py
│   ├── llie_wrapper.py
│   ├── run_n2v_external.py
│   ├── zsn2n.py
│   └── denoise_wrapper.py
│
│
├── downstream_tasks/
│   ├── seg_cell.ipynb
│   ├── llie_wrapper.py
│   ├── segmentation_train.ipynb
│   └── segmentation_eval.ipynb
│
├── data/
│   ├── dataset_creation.ipynb
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

