# Cross-Modality MRI Synthesis with Clinically-Aligned Evaluation

**An Efficient 2D Latent Diffusion Framework for Cross-Modality MRI Synthesis with Clinically-Aligned Evaluation**

Danial Derayati, Hanieh Naderi — University of Tehran, College of Interdisciplinary Science and Technologies

[Paper](#) · [Data (BraTS 2020)](https://www.med.upenn.edu/cbica/brats2020/data.html) · [License](#license)

---

## Overview

This repository contains the implementation for a 2D latent diffusion model (LDM) that synthesizes **FLAIR** MRI images from **T1-contrast-enhanced (T1ce)** scans, trained on the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset using a **single consumer-grade GPU** (NVIDIA RTX 3050, 6GB VRAM).

Most cross-modality MRI synthesis work evaluates outputs using only pixel-similarity metrics (PSNR, SSIM), which correlate poorly with diagnostic quality. This project introduces **task-based clinical detectability metrics** — gCNR, gSNR, and d′ (per AAPM TG-233) — alongside conventional metrics, and benchmarks the LDM against GAN and Transformer baselines trained under identical, controlled conditions.

## Key Results

| Method | PSNR↑ | SSIM↑ | gCNR↑ (retention) | d′/gSNR↑ (retention) |
|---|---|---|---|---|
| GAN | 18.47 ± 2.41 | 0.5765 ± 0.0558 | 0.4783 ± 0.1063 (79.4%) | 1.8376 ± 0.4704 (75.6%) |
| TransUNet | **20.39 ± 2.89** | **0.6798 ± 0.0613** | 0.5013 ± 0.1028 (83.3%) | 1.9408 ± 0.4771 (79.9%) |
| **LDM (proposed, final)** | 15.28 ± 2.56 | 0.4999 ± 0.0763 | **0.5258 ± 0.1334 (87.3%)** | **2.0767 ± 0.6382 (85.4%)** |

The proposed LDM trails the baselines on raw pixel-fidelity metrics but achieves the **strongest clinical detectability scores** — i.e., it best preserves tumor-to-background contrast, the property most relevant to diagnostic use. See the paper for the full comparison, ablations, and statistical tests.

## What's in This Repository

```
├── data/                  # BraTS 2020 preprocessing scripts (does NOT include raw data)
│   ├── preprocess.py      # Cropping, normalization, patient-level splitting
│   └── dataset.py         # PyTorch Dataset / DataLoader definitions
├── models/
│   ├── vae.py             # AutoencoderKL (MONAI) definition and training
│   ├── ldm.py             # Conditional DiffusionModelUNet (MONAI) + DDPM/DDIM
│   ├── gan_baseline.py     # Pix2Pix-style conditional GAN
│   └── transformer_baseline.py  # U-Net + self-attention bottleneck
├── training/
│   ├── train_vae.py
│   ├── train_ldm.py
│   ├── train_gan.py
│   └── train_transformer.py
├── evaluation/
│   ├── metrics.py          # PSNR, SSIM, LPIPS, MAE, gCNR, gSNR, d′
│   ├── fair_comparison.py  # Reproduces Table 2 (main comparison)
│   ├── schedule_ablation.py # Reproduces Tables 4 & 7 (noise-schedule ablation)
│   └── conditioning_sensitivity.py # Reproduces Tables 5 & 6 (wrong-patient test)
├── configs/                # YAML configs for each experiment
├── results/                # Output figures, tables, and generated samples
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python ≥3.9, PyTorch ≥2.0, [MONAI](https://monai.io/) (`generative` package), NumPy, SciPy, scikit-image, matplotlib.

## Data

This project uses the **BraTS 2020** dataset, which is **not included** in this repository due to licensing terms. Register and download it from the official source:

👉 https://www.med.upenn.edu/cbica/brats2020/data.html

After downloading, update the data path in `configs/data_config.yaml` and run:


Default inference configuration: DDIM sampling, 200 steps, classifier-free guidance scale = 2.6, η = 1.0 (identified via a 9-configuration sweep — see paper Section 4).

## Reproducibility Notes

- All models are trained on an identical patient-level train/validation split (seed=42, validation fraction=0.1).
- The main comparison (Table 2) evaluates all methods on the same 43 held-out validation slices.
- A normalization mismatch between VAE training and latent construction (min-max vs. percentile-clipped) was identified and corrected during development — see paper Section 5.5. Preprocessing in this repo uses the corrected, percentile-clipped pipeline.
- Multi-seed validation (seeds 42, 123, 2024) is provided for the schedule-comparison ablation only (Section 5.6), not the main comparison.


## License

This project is licensed under the [MIT License](LICENSE).

Note: this license covers the **code** in this repository only. The BraTS 2020 dataset is governed by its own separate license and usage terms set by the dataset organizers.

## Acknowledgements

Built with [MONAI](https://monai.io/) and [PyTorch](https://pytorch.org/). Data provided by the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) challenge organizers.

## Contact

Danial Derayati — danial.derayati79@gmail.com
