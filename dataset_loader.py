#!/usr/bin/env python3
\"\"\"
PyTorch Dataset & DataLoader utility for the 15-Breed Cattle Classification Dataset.
Supports loading by split (train / val / test) or by federated client ID.
\"\"\"

import os
from pathlib import Path
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object

class CattleBreedDataset(Dataset):
    \"\"\"
    PyTorch Dataset for Indian Cattle & Buffalo Breeds (15-Class).
    
    Args:
        manifest_path (str or Path): Path to split_manifest.csv
        dataset_root (str or Path): Path to master_dataset root folder
        split (str): 'train', 'val', or 'test'
        client_id (str, optional): Client ID if loading a specific federated partition (e.g. 'client_1')
        transform (callable, optional): torchvision transforms to apply to images
    \"\"\"
    def __init__(self, manifest_path=\"Master-Dataset/master_dataset/split_manifest.csv\", 
                 dataset_root=\"Master-Dataset/master_dataset\", 
                 split=\"train\", 
                 client_id=None, 
                 transform=None):
        if torch is None:
            raise ImportError(\"PyTorch and torchvision are required. Install via: pip install torch torchvision\")
        
        self.dataset_root = Path(dataset_root)
        self.split = split.lower()
        self.transform = transform
        
        # Load split manifest
        df = pd.read_csv(manifest_path)
        
        if \"split\" in df.columns:
            df = df[df[\"split\"] == self.split]
        
        if client_id is not None and \"client_id\" in df.columns:
            df = df[df[\"client_id\"] == client_id]
            
        self.data = df.reset_index(drop=True)
        
        # Class mappings
        self.classes = sorted(self.data[\"breed\"].unique().tolist())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Try relative path or path reconstructed from split/breed/filename
        if \"image_path\" in row and (self.dataset_root / row[\"image_path\"]).exists():
            img_path = self.dataset_root / row[\"image_path\"]
        elif (self.dataset_root / self.split / row[\"breed\"] / row[\"filename\"]).exists():
            img_path = self.dataset_root / self.split / row[\"breed\"] / row[\"filename\"]
        else:
            # Fallback path lookup
            img_path = Path(row.get(\"path\", str(self.dataset_root / self.split / row[\"breed\"] / row[\"filename\"])))
            
        image = Image.open(img_path).convert(\"RGB\")
        label = self.class_to_idx[row[\"breed\"]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_default_transforms(img_size=224):
    \"\"\"Standard ImageNet normalization transforms for train and validation.\"\"\"
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

if __name__ == \"__main__\":
    print(\"[*] Testing dataset loader definition...\")
    print(\"To use in your PyTorch project:\")
    print(\"  from dataset_loader import CattleBreedDataset, get_default_transforms\")
