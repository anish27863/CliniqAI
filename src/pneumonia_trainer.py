import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torchvision.models import EfficientNet_B0_Weights
import logging
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PneumoniaTrainer:
    """Pneumonia Detection Model Trainer"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize trainer with configuration"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = "D:\\CliniqAI\\data\\raw\\pneumonia\\chest_xray"
        self.checkpoint_dir = "D:\\CliniqAI\\models\\checkpoints"
        self.specialist_dir = "D:\\CliniqAI\\models\\specialists\\pneumonia"
        
        # Create directories if not exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.specialist_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Data path: {self.data_root}")
        
    def get_transforms(self):
        """Get data augmentation transforms"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return train_transform, val_transform
    
    def load_data(self, batch_size: int = 64):
        """Load pneumonia dataset"""
        logger.info("Loading pneumonia dataset...")
        
        train_transform, val_transform = self.get_transforms()
        
        # Load train dataset
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, "train"),
            transform=train_transform
        )
        
        # Load val dataset
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, "val"),
            transform=val_transform
        )
        
        # Load test dataset
        test_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, "test"),
            transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Classes: {train_dataset.classes}")
        
        return train_loader, val_loader, test_loader, train_dataset.classes
    
    def build_model(self):
        """Build EfficientNet-B0 model"""
        logger.info("Building EfficientNet-B0 model...")
        
        # Load pretrained model
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Replace classifier for binary classification (Normal/Pneumonia)
        num_classes = 2
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        model = model.to(self.device)
        logger.info(f"Model moved to {self.device}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model, val_loader, criterion):
        """Validate model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int = 20, batch_size: int = 64, lr: float = 0.001):
        """Full training pipeline"""
        logger.info(f"Starting Pneumonia Model Training")
        logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
        
        # Load data
        train_loader, val_loader, test_loader, classes = self.load_data(batch_size)
        
        # Build model
        model = self.build_model()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        # Training loop
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > history['best_val_acc']:
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch + 1
                
                checkpoint_path = os.path.join(
                    self.specialist_dir,
                    f"pneumonia_best_epoch{epoch+1}_acc{val_acc:.1f}.pt"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'classes': classes,
                    'timestamp': datetime.now().isoformat()
                }, checkpoint_path)
                
                logger.info(f"✅ Best model saved: {checkpoint_path}")
            
            scheduler.step()
        
        # Test on test set
        logger.info("Evaluating on test set...")
        test_loss, test_acc = self.validate(model, test_loader, criterion)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Save training history
        history_path = os.path.join(self.specialist_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        logger.info(f"✅ Training complete! Best accuracy: {history['best_val_acc']:.2f}%")
        logger.info(f"✅ History saved: {history_path}")
        
        return model, history


if __name__ == "__main__":
    # Create trainer
    trainer = PneumoniaTrainer()
    
    # Train model
    model, history = trainer.train(
        epochs=20,
        batch_size=64,
        lr=0.001
    )
    
    logger.info("🎉 Pneumonia specialist training complete!")
