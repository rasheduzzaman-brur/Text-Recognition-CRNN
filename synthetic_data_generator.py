import os
import random
import csv
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Create directories for the dataset
os.makedirs('synthetic_dataset/val', exist_ok=True)

# Parameters
num_images = 500
image_size = (150, 32)
font_size = 24
alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Path to font file
font_path = "/home/rashed/own_project/custom_ocr_system/font/times new roman.ttf"  # Update this path if needed

# Generate synthetic dataset
with open('synthetic_dataset/val.csv', mode='w', newline='') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(num_images):
        text = ''.join(random.choices(alphabet, k=random.randint(5, 10)))
        image = Image.new('L', image_size, color=255)  # Create a white image
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Font file not found: {font_path}")
            break
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), text, font=font, fill=0)
        filename = f'{i}_{text}.png'
        image.save(f'synthetic_dataset/val/{filename}')
        writer.writerow({'filename': filename, 'label': text})
