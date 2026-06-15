from datasets import load_dataset
import os
from PIL import Image

os.makedirs('test_images/imagenet', exist_ok=True)

print("Skidanje Tiny ImageNet slika...")
ds = load_dataset('zh-plus/tiny-imagenet', split='valid', streaming=True)

count = 0
for item in ds:
    if count >= 30:
        break
    img = item['image'].convert('RGB')
    # Resize na 256x256 - bolje od 64x64 za kompresiju
    img = img.resize((256, 256), Image.LANCZOS)
    img.save(f'test_images/imagenet/{count+1}.jpg')
    count += 1
    print(f'Skinuta slika {count}/30')

print('Gotovo!')