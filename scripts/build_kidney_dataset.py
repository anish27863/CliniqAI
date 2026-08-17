"""
Build Kidney Stone CT dataset for specialist training.
Handles nested folder structure and creates PyTorch dataset loaders.
Classes: Cyst, Normal, Stone, Tumor
"""

import os
import pandas as pd
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

class KidneyCTDataset(Dataset):
    """PyTorch dataset for Kidney CT classification."""
    
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of class labels
            class_to_idx: Dict mapping class names to indices
            transform: PyTorch transforms to apply
        """
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


def build_kidney_dataloaders(data_root, batch_size=32, num_workers=0, image_size=224, test_size=0.2, val_size=0.1):
    """
    Build train, val, test dataloaders for kidney CT dataset.
    
    Args:
        data_root: Root directory containing kidney_stone folder
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader (use 0 on Windows)
        image_size: Size to resize images to
        test_size: Proportion for test set
        val_size: Proportion for val set (from remaining after test split)
    
    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_names'
    """
    
    data_dir = Path(data_root) / "kidney_stone"
    csv_path = data_dir / "kidneyData.csv"
    image_base_dir = data_dir / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Map class names to indices
    class_names = ["Cyst", "Normal", "Stone", "Tumor"]
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Build image paths
    image_paths = []
    labels = []
    
    for idx, row in df.iterrows():
        class_name = row['Class']
        image_id = row['image_id']
        
        # Construct path based on class
        img_path = image_base_dir / class_name / f"{image_id}.jpg"
        
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(class_to_idx[class_name])
        else:
            logger.warning(f"Image not found: {img_path}")
    
    logger.info(f"✅ Loaded {len(image_paths)} images from CSV")
    
    # Convert to numpy for stratified split
    image_paths = np.array(image_paths)
    labels = np.array(labels, dtype=np.int64)
    
    # Split: test first, then val from remaining
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels
    )
    
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
    train_dataset = KidneyCTDataset(
        train_paths, train_labels, class_to_idx, transform=train_transform
    )
    
    val_dataset = KidneyCTDataset(
        val_paths, val_labels, class_to_idx, transform=val_test_transform
    )
    
    test_dataset = KidneyCTDataset(
        test_paths, test_labels, class_to_idx, transform=val_test_transform
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
    # Test the dataset builder
    dataloaders = build_kidney_dataloaders(
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