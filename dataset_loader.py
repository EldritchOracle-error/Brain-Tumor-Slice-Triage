# dataset_loader.py: Optimized loader using fast directory scanning
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATASET_DIR = r"D:\MDM_Project\Dataset"

# 1. Custom Dataset for preprocessed .npy files
class Brats2DSliceDataset(Dataset):
    def __init__(self, img_paths, seg_paths, is_training=True):
        self.is_training = is_training
        self.img_paths = img_paths
        self.seg_paths = seg_paths
            
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]) # Shape: (4, 128, 128)
        seg = np.load(self.seg_paths[idx]) # Shape: (128, 128)
        
        img = torch.tensor(img, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.long)
        
        # 4-bit availability mask (1 = present, 0 = missing)
        mask = torch.ones(4, dtype=torch.float32)
        
        if self.is_training:
            # Modality Dropout: randomly drop 0 to 3 channels (ensure at least 1 stays)
            if np.random.rand() > 0.2:
                num_drop = np.random.randint(1, 4)
                drop_indices = np.random.choice(4, num_drop, replace=False)
                for d_idx in drop_indices:
                    img[d_idx] = 0.0
                    mask[d_idx] = 0.0
                    
        return img, seg, mask

# Fast Patient Splitter using os.scandir
def get_patient_splits(dataset_dir):
    print("Scanning dataset directory (fast scan)...")
    img_files = []
    seg_files = []
    
    # Fast scan using os.scandir
    with os.scandir(dataset_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith("_img.npy"):
                img_path = entry.path
                seg_path = entry.path.replace("_img.npy", "_mask.npy")
                if os.path.exists(seg_path):
                    img_files.append(img_path)
                    seg_files.append(seg_path)
                    
    print(f"Found total {len(img_files)} slice pairs.")
    if len(img_files) == 0:
        raise ValueError(f"No valid slice pairs found in {dataset_dir}!")
        
    # Extract unique patient IDs
    patient_to_slices = {}
    for img_p in img_files:
        base_name = os.path.basename(img_p)
        # Assuming patient ID is everything before '_slice_'
        p_id = base_name.split("_slice_")[0]
        if p_id not in patient_to_slices:
            patient_to_slices[p_id] = {"img": [], "seg": []}
        patient_to_slices[p_id]["img"].append(img_p)
        patient_to_slices[p_id]["seg"].append(img_p.replace("_img.npy", "_mask.npy"))
        
    patients = sorted(list(patient_to_slices.keys()))
    print(f"Found {len(patients)} unique patients.")
    
    # 70% train, 15% val, 15% test
    train_val, test_p = train_test_split(patients, test_size=0.15, random_state=42)
    train_p, val_p = train_test_split(train_val, test_size=0.1765, random_state=42)
    
    print(f"Patient Splits -> Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")
    
    def gather_paths(p_list):
        imgs, segs = [], []
        for p in p_list:
            imgs.extend(patient_to_slices[p]["img"])
            segs.extend(patient_to_slices[p]["seg"])
        return imgs, segs

    train_imgs, train_segs = gather_paths(train_p)
    val_imgs, val_segs = gather_paths(val_p)
    test_imgs, test_segs = gather_paths(test_p)
    
    return (train_imgs, train_segs), (val_imgs, val_segs), (test_imgs, test_segs)


# 2. ResUNet with Dropout (for MC Dropout) and FiLM Conditioning
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResUNetWithFiLM(nn.Module):
    def __init__(self, in_channels=4, out_classes=4, dropout_p=0.2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64, dropout_p)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128, dropout_p)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256, dropout_p)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(256, 512, dropout_p)
        
        self.film_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 512 * 2)
        )
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = ConvBlock(512, 256, dropout_p)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ConvBlock(256, 128, dropout_p)
        
        self.up0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec0 = ConvBlock(128, 64, dropout_p)
        
        self.out_conv = nn.Conv2d(64, out_classes, 1)

    def forward(self, x, mask):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        film_params = self.film_fc(mask)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        b = b * (1 + gamma) + beta
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)
        d0 = torch.cat([d0, e1], dim=1)
        d0 = self.dec0(d0)
        
        return self.out_conv(d0)