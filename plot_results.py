import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ─── Rezultati ────────────────────────────────────────────────────────────────

kodak = {
    "bpp":     [0.02,  0.04,  0.06,  0.09,  0.12],
    "PSNR":    [19.830, 21.402, 23.026, 24.406, 24.292],
    "MS-SSIM": [0.6709, 0.7538, 0.8180, 0.8667, 0.8899],
    "LPIPS":   [0.3429, 0.2516, 0.1764, 0.1277, 0.1062],
    "DISTS":   [0.2024, 0.1547, 0.1100, 0.0793, 0.0659],
}

imagenet = {
    "bpp":     [0.02,  0.04,  0.06,  0.09,  0.12],
    "PSNR":    [18.693, 20.165, 22.414, 24.429, 23.353],
    "MS-SSIM": [0.7580, 0.8309, 0.8845, 0.9194, 0.9318],
    "LPIPS":   [0.2676, 0.1890, 0.1276, 0.0895, 0.0788],
    "DISTS":   [0.1915, 0.1512, 0.1092, 0.0824, 0.0723],
}

os.makedirs("plots_new", exist_ok=True)

metrics = [
    ("PSNR",    "PSNR (dB) ↑",    False),
    ("MS-SSIM", "MS-SSIM ↑",      False),
    ("LPIPS",   "LPIPS ↓",        True),
    ("DISTS",   "DISTS ↓",        True),
]

# ─── Kombinovani graf (2 reda x 4 kolone) ──────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(19, 9))
fig.suptitle("DiffEIC — Rate-Distortion i Rate-Perception krivulje", fontsize=14, fontweight='bold')

datasets = [("Kodak", kodak), ("ImageNet (Unsplash)", imagenet)]

for row, (ds_name, data) in enumerate(datasets):
    for col, (metric, ylabel, lower_better) in enumerate(metrics):
        ax = axes[row][col]
        ax.plot(data["bpp"], data[metric],
                marker='o', linewidth=2, markersize=6,
                color='#e74c3c', label='DiffEIC (Ours)')
        ax.set_xlabel("bpp", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric} on {ds_name}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        if lower_better:
            ax.invert_yaxis()

plt.tight_layout()
plt.savefig("plots_new/rate_distortion.png", dpi=150, bbox_inches='tight')
print("Grafovi sačuvani u: plots_new/rate_distortion.png")

# ─── Pojedinacni grafovi za svaki dataset (1 red x 4 kolone) ──────────────────
for ds_name, data in datasets:
    fig2, axes2 = plt.subplots(1, 4, figsize=(19, 4))
    fig2.suptitle(f"DiffEIC na {ds_name} datasetu", fontsize=13, fontweight='bold')

    for col, (metric, ylabel, lower_better) in enumerate(metrics):
        ax = axes2[col]
        ax.plot(data["bpp"], data[metric],
                marker='o', linewidth=2.5, markersize=7,
                color='#e74c3c', label='DiffEIC (Ours)')
        ax.set_xlabel("bpp", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric} on {ds_name}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        if lower_better:
            ax.invert_yaxis()

    plt.tight_layout()
    fname = ds_name.split()[0].lower()
    plt.savefig(f"plots_new/rate_distortion_{fname}.png", dpi=150, bbox_inches='tight')
    print(f"Sačuvano: plots_new/rate_distortion_{fname}.png")

print("Gotovo!")