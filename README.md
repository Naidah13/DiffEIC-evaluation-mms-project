# DiffEIC Evaluation — MMS 2025/26 Project

This repository contains the modified inference pipeline, evaluation scripts,
and results used to reproduce and evaluate the DiffEIC framework on the Kodak
dataset and a custom ImageNet-style subset, as part of the MMS 2025/26 course
project at the Faculty of Electrical Engineering, University of Sarajevo.

## Original Work

This project builds upon the official DiffEIC implementation:

> Z. Li, Y. Zhou, H. Wei, C. Ge, and J. Jiang, "Toward Extreme Image
> Compression with Latent Feature Guidance and Diffusion Prior,"
> IEEE Transactions on Circuits and Systems for Video Technology,
> vol. 35, no. 1, pp. 888-899, 2025. doi: 10.1109/TCSVT.2024.3455576

Original repository: https://github.com/huai-chang/DiffEIC

Please cite the original authors if you use this work:

```bibtex
@article{li2024towards,
  author={Li, Zhiyuan and Zhou, Yanhui and Wei, Hao and Ge, Chenyang and Jiang, Jingwen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Toward Extreme Image Compression with Latent Feature Guidance and Diffusion Prior},
  year={2025},
  volume={35},
  number={1},
  pages={888-899},
  doi={10.1109/TCSVT.2024.3455576}}
```

## Repository Structure

```
├── inference_partition.py       # Modified inference script (xformers fix)
├── download_imagenet.py         # Downloads/prepares the ImageNet-style subset
├── resize_images.py             # Resizes images to 512x512 for inference
├── evaluate.py                  # PSNR / MS-SSIM / LPIPS / DISTS on Kodak
├── evaluate_imagenet.py          # PSNR / MS-SSIM / LPIPS / DISTS on ImageNet subset
├── compute_bpp.py               # Computes average bpp from bitstream files
├── plot_results.py              # Generates rate-distortion/perception plots
├── requirements.txt             # Python dependencies
├── requirements_no_numpy.txt    # Dependencies without numpy (Windows build fix)
├── test_images/                 # Input images (Kodak + ImageNet subset)
├── rezultati/                   # Reconstructions, Kodak, lambda=4 (0.06 bpp)
├── rezultati_1_2_1/              # Reconstructions, Kodak, lambda=1 (0.12 bpp)
├── rezultati_1_2_2/              # Reconstructions, Kodak, lambda=2 (0.09 bpp)
├── rezultati_1_2_8/              # Reconstructions, Kodak, lambda=8 (0.04 bpp)
├── rezultati_1_2_16/             # Reconstructions, Kodak, lambda=16 (0.02 bpp)
├── rezultati_imagenet_1_2_1/      # Reconstructions, ImageNet, lambda=1
├── rezultati_imagenet_1_2_2/      # Reconstructions, ImageNet, lambda=2
├── rezultati_imagenet_1_2_4/      # Reconstructions, ImageNet, lambda=4
├── rezultati_imagenet_1_2_8/      # Reconstructions, ImageNet, lambda=8
├── rezultati_imagenet_1_2_16/     # Reconstructions, ImageNet, lambda=16
└── plots_new/                    # Final rate-distortion/perception plots (with DISTS)
```

## Modifications Made

1. **`inference_partition.py`**: Modified to unconditionally call
   `disable_xformers()` regardless of device, since xformers was not
   available in our environment.

2. **`download_imagenet.py`**: New script used to prepare the ImageNet-style
   evaluation subset (30 images, resized to 512x512). Due to access
   restrictions on the official ImageNet-1k dataset (institutional
   verification required), this subset consists of diverse natural images
   sourced from Unsplash, covering categories representative of ImageNet
   content (architecture, nature, animals, portraits, interiors).

3. **`resize_images.py`**: Resizes input images to 512x512 to fit within
   the 6GB VRAM budget of the evaluation GPU.

4. **`evaluate.py` / `evaluate_imagenet.py`**: Evaluation scripts computing
   PSNR, MS-SSIM, LPIPS, and DISTS between original and reconstructed images
   for the Kodak and ImageNet datasets, respectively.

5. **`compute_bpp.py`**: Computes average bits-per-pixel from saved
   bitstream files, used to recover exact bpp values for configurations
   where the inference log was not retained.

6. **`plot_results.py`**: Generates rate-distortion and rate-perception
   plots (PSNR, MS-SSIM, LPIPS, DISTS vs. bpp) for both datasets, saved to
   `plots_new/`.

## Experimental Setup

- GPU: NVIDIA RTX 4050 Laptop GPU (6GB VRAM) vs. original RTX 4090 (24GB VRAM)
- Denoising steps: 20 (instead of 50)
- Sampler: DDPM
- Datasets:
  - **Kodak** (24 images, 768x512) — standard benchmark, same as used by the
    original authors
  - **Custom ImageNet-style subset** (30 images, 512x512, sourced from
    Unsplash) — not used in the original DiffEIC evaluation, providing an
    independent test of generalisation
- Bitrate configurations: λ ∈ {1, 2, 4, 8, 16}, corresponding to
  approximately 0.12, 0.09, 0.06, 0.04, and 0.02 bpp

## Results Summary

| λ | bpp (Kodak) | PSNR | MS-SSIM | LPIPS | DISTS | bpp (ImageNet) | PSNR | MS-SSIM | LPIPS | DISTS |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0203 | 19.83 | 0.671 | 0.343 | 0.202 | 0.0203 | 18.69 | 0.758 | 0.268 | 0.192 |
| 8  | 0.0358 | 21.40 | 0.754 | 0.252 | 0.155 | 0.0358 | 20.17 | 0.831 | 0.189 | 0.151 |
| 4  | 0.0568 | 23.03 | 0.818 | 0.176 | 0.110 | 0.0568 | 22.41 | 0.885 | 0.128 | 0.109 |
| 2  | 0.0867 | 24.41 | 0.867 | 0.128 | 0.079 | 0.0867 | 24.43 | 0.919 | 0.090 | 0.082 |
| 1  | 0.1233 | 24.29 | 0.890 | 0.106 | 0.066 | 0.1184 | 23.35 | 0.932 | 0.079 | 0.072 |

Full discussion, figures, and analysis are available in the accompanying
project report.

## Setup

```bash
conda create -n diffeic python=3.8
conda activate diffeic
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_no_numpy.txt --prefer-binary
```

Download the Stable Diffusion 2.1-base checkpoint and DiffEIC pretrained
weights (lambda configurations 1/2/4/8/16) as described in the
[original repository](https://github.com/huai-chang/DiffEIC).

## Usage

```bash
# Run inference for a given lambda configuration
python inference_partition.py --ckpt_sd ./weight/v2-1_512-ema-pruned.ckpt \
    --ckpt_lc ./weight/1_2_4/lc.ckpt --config configs/model/diffeic.yaml \
    --input ./test_images/kodak --output ./rezultati --steps 20 --device cuda

# Evaluate results
python evaluate.py
python evaluate_imagenet.py

# Generate plots
python plot_results.py
```

## Authors

- Naida Hasović — nhasovic1@etf.unsa.ba
- Lana Malinov — lmalinov1@etf.unsa.ba
