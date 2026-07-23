"""
Extract CIFAR-10 pickle files to PNG images for router training.
"""

import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import random

random.seed(42)

CIFAR_DIR = Path("./data/raw/cifar10/cifar-10-batches-py")
OUTPUT_DIR = Path("./data/router_training/non_medical")
SAMPLES_NEEDED = 300

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_cifar_batch(batch_file):
    """Load a CIFAR-10 batch file."""
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    return batch['data'], batch['labels']

def cifar_to_image(cifar_array):
    """Convert CIFAR-10 32x32 array to PIL Image."""
    # CIFAR stores as (3072,) array: reshape to (32, 32, 3)
    img_array = cifar_array.reshape(3, 32, 32)
    # Convert from (C, H, W) to (H, W, C)
    img_array = np.transpose(img_array, (1, 2, 0))
    return Image.fromarray(img_array.astype(np.uint8))

print("Extracting CIFAR-10 images...")

all_images = []

# Load training batches
for batch_num in range(1, 6):
    batch_file = CIFAR_DIR / f"data_batch_{batch_num}"
    if batch_file.exists():
        data, labels = load_cifar_batch(batch_file)
        for img_array in data:
            all_images.append(img_array)

print(f"Loaded {len(all_images)} images from CIFAR-10")

# Sample randomly
random.shuffle(all_images)
sampled = all_images[:SAMPLES_NEEDED]

# Save as PNG
for i, img_array in enumerate(sampled):
    img = cifar_to_image(img_array)
    img.save(OUTPUT_DIR / f"non_medical_{i:04d}.png")

print(f"✅ Extracted {len(sampled)} images to {OUTPUT_DIR}")