"""
Kidney Stone Specialist Model Training - CT Classification
EfficientNet-B0 fine-tuning with early stopping and best model checkpoint.
Classes: Cyst, Normal, Stone, Tumor
"""

import os
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import dataset builder
sys.path.insert(0, str(Path.cwd() / "scripts"))
from build_kidney_dataset import build_kidney_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "model_name": "efficientnet_b0",
    "data_root": "./data/raw",
    "checkpoint_dir": "./models/checkpoints",
    "log_dir": "./logs",
    "batch_size": 32,  # Adjusted for RTX 4050 (6GB VRAM)
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 10,  # Early stopping patience
    "image_size": 224,
    "num_workers": 0,  # Windows: always 0
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create required directories."""
    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Directories ready")


def get_device():
    """Get device and log GPU info."""
    device = CONFIG["device"]
    if device == "cuda":
        logger.info(f"🚀 Using CUDA")
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info(f"⚠️  Using CPU")
    return device


# ============================================================================
# MODEL
# ============================================================================

def build_model(num_classes=4, device="cuda"):
    """Build EfficientNet-B0 kidney specialist."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    
    logger.info(f"🔨 Building EfficientNet-B0 specialist...")
    
    # Load pretrained weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze backbone layers (optional, can be unfrozen later)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    logger.info(f"✅ Model ready on {device}")
    return model


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = running_loss / len(train_loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """Validate on val set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    avg_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_acc, all_preds, all_labels


def test(model, test_loader, device):
    """Test on test set with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("KIDNEY STONE SPECIALIST TRAINING - CT CLASSIFICATION")
    logger.info("=" * 70)
    
    # Setup
    setup_directories()
    device = get_device()
    
    # Load data
    logger.info("\n📊 Loading dataset...")
    dataloaders = build_kidney_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        image_size=CONFIG["image_size"]
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    class_names = dataloaders["class_names"]
    num_classes = dataloaders["num_classes"]
    
    # Build model
    logger.info("\n🏗️  Building model...")
    model = build_model(num_classes=num_classes, device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )
    
    # Training state
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Training loop
    logger.info("\n🚀 Starting training...\n")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        logger.info(
            f"Epoch {epoch:3d}/{CONFIG['num_epochs']} | "
            f"TL: {train_loss:.4f} | TA: {train_acc:.4f} | "
            f"VL: {val_loss:.4f} | VA: {val_acc:.4f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = Path(CONFIG["checkpoint_dir"]) / "kidney_specialist_v1.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✅ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= CONFIG["patience"]:
            logger.info(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            break
    
    # Load best model and test
    logger.info("\n📋 Testing on test set...")
    checkpoint_path = Path(CONFIG["checkpoint_dir"]) / "kidney_specialist_v1.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    test_results = test(model, test_loader, device)
    
    logger.info(
        f"\n{'='*70}\n"
        f"TEST RESULTS\n"
        f"{'='*70}\n"
        f"Accuracy:  {test_results['accuracy']:.4f}\n"
        f"Precision: {test_results['precision']:.4f}\n"
        f"Recall:    {test_results['recall']:.4f}\n"
        f"F1 Score:  {test_results['f1']:.4f}\n"
    )
    
    # Save metrics
    metrics = {
        "model": CONFIG["model_name"],
        "task": "Kidney Stone (CT Classification)",
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_results["accuracy"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "checkpoint": str(checkpoint_path),
        "class_names": class_names
    }
    
    metrics_path = Path(CONFIG["log_dir"]) / "kidney_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Metrics saved to {metrics_path}")
    
    # Plot training history
    logger.info("\n📈 Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Kidney Stone Specialist - Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history["train_acc"], label="Train Accuracy", marker="o", markersize=3)
    ax2.plot(history["val_acc"], label="Val Accuracy", marker="s", markersize=3)
    ax2.axhline(y=best_val_acc, color="g", linestyle="--", label=f"Best Val: {best_val_acc:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Kidney Stone Specialist - Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plot_path = Path(CONFIG["log_dir"]) / "kidney_training_history.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info(f"✅ Plot saved to {plot_path}")
    plt.close()
    
    # Plot confusion matrix
    logger.info("📊 Plotting confusion matrix...")
    cm = test_results["confusion_matrix"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"}
    )
    plt.title("Kidney Stone Specialist - Test Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    cm_path = Path(CONFIG["log_dir"]) / "kidney_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    logger.info(f"✅ Confusion matrix saved to {cm_path}")
    plt.close()
    
    logger.info("\n" + "="*70)
    logger.info("✅ KIDNEY STONE SPECIALIST TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()