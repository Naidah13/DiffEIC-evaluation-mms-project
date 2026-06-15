import os

DATA_DIR = "./rezultati_imagenet_1_2_1/data"
WIDTH = 512
HEIGHT = 512

total_bpp = []

files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]

# Grupiramo fajlove po imenu slike (stem) - moze biti vise streamova po slici
from collections import defaultdict
groups = defaultdict(int)

for f in files:
    stem = os.path.splitext(f)[0]
    size_bytes = os.path.getsize(os.path.join(DATA_DIR, f))
    groups[stem] += size_bytes

for stem, total_bytes in groups.items():
    bpp = (total_bytes * 8) / (WIDTH * HEIGHT)
    total_bpp.append(bpp)
    print(f"{stem}: {total_bytes} bytes -> {bpp:.4f} bpp")

print("-" * 40)
print(f"Broj slika: {len(total_bpp)}")
print(f"avg bpp: {sum(total_bpp) / len(total_bpp)}")