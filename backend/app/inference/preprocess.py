"""
Shared inference-time transform. Every model in this pipeline (both router
stages and all 7 specialists) was trained on 224x224, ImageNet-normalized
input -- so one transform serves the whole pipeline, applied fresh at each
stage since PIL images get consumed by ToTensor().
"""

from torchvision import transforms
from app.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
