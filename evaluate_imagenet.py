import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
import lpips
import pyiqa

# ─── Config ───────────────────────────────────────────────────────────────────
ORIGINAL_DIR = "./test_images/imagenet_512"

RESULTS = {
    "0.02 bpp": "./rezultati_imagenet_1_2_16",
    "0.04 bpp": "./rezultati_imagenet_1_2_8",
    "0.06 bpp": "./rezultati_imagenet_1_2_4",
    "0.09 bpp": "./rezultati_imagenet_1_2_2",
    "0.12 bpp": "./rezultati_imagenet_1_2_1",
}
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='alex').to(device)
dists_fn = pyiqa.create_metric('dists', device=device)
to_tensor = transforms.ToTensor()

def load_image(path):
    return Image.open(path).convert("RGB")

def psnr(img1, img2):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_msssim(img1, img2):
    t1 = to_tensor(img1).unsqueeze(0).to(device)
    t2 = to_tensor(img2).unsqueeze(0).to(device)
    if t1.shape[-1] < 160 or t1.shape[-2] < 160:
        return float('nan')
    return ms_ssim(t1, t2, data_range=1.0).item()

def compute_lpips(img1, img2):
    t1 = to_tensor(img1).unsqueeze(0).to(device) * 2 - 1
    t2 = to_tensor(img2).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        return loss_fn(t1, t2).item()

def compute_dists(img1, img2):
    t1 = to_tensor(img1).unsqueeze(0).to(device)
    t2 = to_tensor(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        return dists_fn(t1, t2).item()

print(f"{'Bitrate':<12} {'PSNR':>8} {'MS-SSIM':>10} {'LPIPS':>8} {'DISTS':>8}")
print("-" * 52)

for label, result_dir in sorted(RESULTS.items()):
    psnr_vals, msssim_vals, lpips_vals, dists_vals = [], [], [], []

    orig_files = sorted([f for f in os.listdir(ORIGINAL_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for fname in orig_files:
        orig_path = os.path.join(ORIGINAL_DIR, fname)
        stem = os.path.splitext(fname)[0]
        rec_path = os.path.join(result_dir, stem + ".png")

        if not os.path.exists(rec_path):
            continue

        orig = load_image(orig_path)
        rec  = load_image(rec_path)

        if orig.size != rec.size:
            rec = rec.resize(orig.size, Image.LANCZOS)

        psnr_vals.append(psnr(orig, rec))
        msssim_vals.append(compute_msssim(orig, rec))
        lpips_vals.append(compute_lpips(orig, rec))
        dists_vals.append(compute_dists(orig, rec))

    avg_psnr   = np.nanmean(psnr_vals)
    avg_msssim = np.nanmean(msssim_vals)
    avg_lpips  = np.nanmean(lpips_vals)
    avg_dists  = np.nanmean(dists_vals)

    print(f"{label:<12} {avg_psnr:>8.3f} {avg_msssim:>10.4f} {avg_lpips:>8.4f} {avg_dists:>8.4f}")

print("-" * 52)
print("Gotovo!")