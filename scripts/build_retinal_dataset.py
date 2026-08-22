"""
Build Retinal (Diabetic Retinopathy) dataset for specialist training.
Handles nested folder structure and creates PyTorch dataset loaders.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetinalDRDataset(Dataset):
    """PyTorch dataset for Diabetic Retinopathy classification."""
    
    def __init__(self, image_dir, csv_path, transform=None):
        """
        Args:
            image_dir: Path to folder containing images (handles nested structure)
            csv_path: Path to CSV with id_code and diagnosis columns
            transform: PyTorch transforms to apply
        """
        self.csv = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Handle nested folder structure
        self.actual_image_dir = self.image_dir
        for subdir in self.image_dir.glob("*"):
            if subdir.is_dir() and subdir.name in ["train_images", "val_images", "test_images"]:
                self.actual_image_dir = subdir
                break
        
        logger.info(f"Loading images from: {self.actual_image_dir}")
        logger.info(f"Dataset size: {len(self.csv)} samples")
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = self.actual_image_dir / img_name
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        label = int(row['diagnosis'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_retinal_dataloaders(data_root, batch_size=32, num_workers=0, image_size=224):
    """
    Build train, val, test dataloaders for retinal DR dataset.
    
    Args:
        data_root: Root directory containing diabetic_retinopathy folder
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader (use 0 on Windows)
        image_size: Size to resize images to
    
    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_names'
    """
    
    data_dir = Path(data_root) / "diabetic_retinopathy"
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
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
    
    # Create datasets
    train_dataset = RetinalDRDataset(
        image_dir=data_dir / "train_images",
        csv_path=data_dir / "train_1.csv",
        transform=train_transform
    )
    
    val_dataset = RetinalDRDataset(
        image_dir=data_dir / "val_images",
        csv_path=data_dir / "valid.csv",
        transform=val_test_transform
    )
    
    test_dataset = RetinalDRDataset(
        image_dir=data_dir / "test_images",
        csv_path=data_dir / "test.csv",
        transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    
    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Val samples: {len(val_dataset)}")
    logger.info(f"✅ Test samples: {len(test_dataset)}")
    logger.info(f"✅ Classes: {class_names}")
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names,
        "num_classes": 5
    }


if __name__ == "__main__":
    # Test the dataset builder
    dataloaders = build_retinal_dataloaders(
        data_root="./data/raw",
        batch_size=32,
        num_workers=0,
        image_size=224
    )
    
    # Verify one batch
    train_loader = dataloaders["train"]
    images, labels = next(iter(train_loader))
    
    logger.info(f"\n✅ Dataset builder working!")
    logger.info(f"Batch shape: {images.shape}")
    logger.info(f"Label shape: {labels.shape}")
    logger.info(f"Label values: {labels.unique().tolist()}")