# Cross-Modality MRI Synthesis with Clinically-Aligned Evaluation

**An Efficient 2D Latent Diffusion Framework for Cross-Modality MRI Synthesis with Clinically-Aligned Evaluation**



This repository accompanies the manuscript submitted to *Computerized Medical Imaging and Graphics*. It provides the code used to produce all results, tables, and figures reported in the paper, in support of reproducibility during review.

---

## Overview

This work presents a 2D latent diffusion model (LDM) that synthesizes **FLAIR** MRI images from **T1-contrast-enhanced (T1ce)** scans, trained on the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset using a single consumer-grade GPU (NVIDIA RTX 3050, 6GB VRAM). The LDM is compared against GAN and Transformer baselines trained under identical, controlled conditions.

Rather than relying solely on pixel-similarity metrics (PSNR, SSIM), which correlate poorly with diagnostic quality, this work introduces **task-based clinical detectability metrics** — gCNR, gSNR, and d′ (per AAPM TG-233) — to the evaluation protocol.

## Key Result

On the common fair evaluation set (n = 43 validation slices), the proposed LDM achieves the strongest clinical detectability performance among the compared methods:

- **gCNR** = 0.5258 ± 0.1334 (87.3% retention relative to ground truth)
- **d′/gSNR** = 2.0767 ± 0.6382 (85.4% retention)

while GAN and Transformer baselines score higher on conventional pixel-fidelity metrics (PSNR, SSIM, LPIPS). Full results, ablations, and statistical tests are reported in the manuscript (Sections 5.1–5.6).

## Samples of results
Figures are available in final pics
## Repository Contents

This repository contains the full implementation used in the study, including:

- Data preprocessing for BraTS 2020 (cropping, normalization, patient-level splitting)
- The variational autoencoder (VAE) and conditional latent diffusion model (DDPM/DDIM)
- The GAN (pix2pix-style) and Transformer baseline implementations
- Evaluation code for all reported metrics (PSNR, SSIM, LPIPS, MAE, gCNR, gSNR, d′)
- Scripts reproducing the noise-schedule ablation and the conditioning-sensitivity experiment

## Data

This project uses the **BraTS 2020** dataset, which is **not redistributed** in this repository due to its licensing terms. It is available directly from the official source:

👉 https://www.med.upenn.edu/cbica/brats2020/data.html

No new human-subject data were collected for this study; all imaging data are the publicly available, de-identified BraTS 2020 release.

## Reproducibility Notes

- All three models (LDM, GAN, Transformer) are trained and evaluated on an identical patient-level train/validation split (seed = 42, validation fraction = 0.1).
- The main comparison (Table 2 in the manuscript) evaluates all methods on the same held-out validation slices, using the same preprocessing and metric implementation.
- A normalization inconsistency identified during development (between VAE training and latent construction) was corrected prior to final results; this is documented in the manuscript (Section 5.5) for transparency.
- Multi-seed validation (three random seeds) is reported for the noise-schedule ablation specifically (Section 5.6), to confirm result stability.



## License

This project is licensed under the [MIT License](LICENSE).

Note: this license covers the **code** in this repository only. The BraTS 2020 dataset is governed by its own separate license and usage terms set by the dataset organizers.

## Contact

Danial Derayati — danial.derayati79@gmail.com
