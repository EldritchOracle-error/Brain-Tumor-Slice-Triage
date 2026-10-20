# train.py: Training Loop with Dice + Focal Loss and Mixed Precision
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset_loader import DATASET_DIR, Brats2DSliceDataset, get_patient_splits, ResUNetWithFiLM

CHECKPOINT_DIR = "./brats_project/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Combined Loss: Dice Loss + Focal Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, num_classes=4):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - torch.mean(dice)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

class BratsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.focal(logits, targets)


# 2. Training Function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Get patient splits
    train_data, val_data, test_data = get_patient_splits(DATASET_DIR)
    
    train_dataset = Brats2DSliceDataset(train_data[0], train_data[1], is_training=True)
    val_dataset = Brats2DSliceDataset(val_data[0], val_data[1], is_training=False)
    
    print(f"Train dataset samples: {len(train_dataset)} | Val dataset samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    model = ResUNetWithFiLM(in_channels=4, out_classes=4, dropout_p=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = BratsLoss()
    scaler = GradScaler()
    
    epochs = 10 
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for imgs, segs, masks in loop:
            imgs, segs, masks = imgs.to(device), segs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(imgs, masks)
                loss = criterion(outputs, segs)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, segs, masks in val_loader:
                imgs, segs, masks = imgs.to(device), segs.to(device), masks.to(device)
                outputs = model(imgs, masks)
                loss = criterion(outputs, segs)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model checkpoint to {ckpt_path}")

    print("Training complete!")

if __name__ == "__main__":
    train_model()