import os
from PIL import Image

input_dir = 'test_images/imagenet'
output_dir = 'test_images/imagenet_512'
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for i, fname in enumerate(files):
    img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    img.save(os.path.join(output_dir, fname))
    print(f'Resized {i+1}/{len(files)}: {fname}')

print('Gotovo!')