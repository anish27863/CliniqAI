"""
Build Brain Tumor MRI dataset for specialist training.
Standard folder-per-class structure (Training/Testing already split).
Classes: glioma, meningioma, notumor, pituitary
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    """PyTorch dataset for Brain Tumor MRI classification."""

    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

        logger.info(f"Dataset size: {len(self.image_paths)} samples")
        logger.info(f"Classes: {list(class_to_idx.keys())}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def _collect_images(split_dir, class_names, class_to_idx):
    """Walk a Training/ or Testing/ folder and collect (path, label) pairs."""
    image_paths = []
    labels = []

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Missing class folder: {class_dir}")
            continue

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])

    return image_paths, labels


def build_brain_tumor_dataloaders(data_root, batch_size=32, num_workers=0, image_size=224, val_size=0.1):
    """
    Build train, val, test dataloaders for brain tumor MRI dataset.

    Args:
        data_root: Root directory containing brain_tumor folder
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader (use 0 on Windows)
        image_size: Size to resize images to
        val_size: Proportion of Training data held out for validation

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_names'
    """

    data_dir = Path(data_root) / "brain_tumor"
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    # Collect all training images, then split into train/val
    all_train_paths, all_train_labels = _collect_images(train_dir, class_names, class_to_idx)
    test_paths, test_labels = _collect_images(test_dir, class_names, class_to_idx)

    logger.info(f"✅ Loaded {len(all_train_paths)} training images, {len(test_paths)} test images")

    all_train_paths = np.array(all_train_paths)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels, dtype=np.int64)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths, all_train_labels,
        test_size=val_size, random_state=42, stratify=all_train_labels
    )

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = BrainTumorDataset(train_paths, train_labels, class_to_idx, transform=train_transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, class_to_idx, transform=val_test_transform)
    test_dataset = BrainTumorDataset(test_paths, test_labels, class_to_idx, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )

    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Val samples: {len(val_dataset)}")
    logger.info(f"✅ Test samples: {len(test_dataset)}")
    logger.info(f"✅ Classes: {class_names}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names,
        "num_classes": 4
    }


if __name__ == "__main__":
    dataloaders = build_brain_tumor_dataloaders(
        data_root="./data/raw",
        batch_size=32,
        num_workers=0,
        image_size=224
    )

    train_loader = dataloaders["train"]
    images, labels = next(iter(train_loader))

    logger.info(f"\n✅ Dataset builder working!")
    logger.info(f"Batch shape: {images.shape}")
    logger.info(f"Label shape: {labels.shape}")
    logger.info(f"Label values: {labels.unique().tolist()}")