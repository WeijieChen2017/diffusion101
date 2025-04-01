import os
import json
from typing import List, Dict, Optional
import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandRotate90d,
    RandFlipd,
)

def prepare_dataset(
    data_div_json: str,
    output_dir: str,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    batch_size: int = 1,
    shuffle: bool = True,
    train: bool = True
) -> DataLoader:
    """
    Prepare dataset from preprocessed NPY files using MONAI's CacheDataset
    
    Args:
        data_div_json: Path to data division JSON file
        output_dir: Directory containing preprocessed NPY files
        cache_rate: Rate of data to cache (0.0 to 1.0)
        num_workers: Number of workers for data loading
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        train: Whether this is for training (True) or validation (False)
    """
    # Load data division
    with open(data_div_json, 'r') as f:
        data_div = json.load(f)
    
    # Select appropriate split
    split_key = 'train' if train else 'val'
    case_list = data_div[split_key]
    
    # Prepare data list for MONAI
    data_list = []
    for case_name in case_list:
        ct_dir = os.path.join(output_dir, 'LDM_CT_slices', case_name)
        sct_dir = os.path.join(output_dir, 'LDM_sCT_slices', case_name)
        
        # Get all slice files
        slice_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.npy')])
        
        for slice_file in slice_files:
            data_list.append({
                'CT': os.path.join(ct_dir, slice_file),
                'sCT': os.path.join(sct_dir, slice_file)
            })
    
    # Define base transforms
    base_transforms = [
        LoadImaged(keys=['CT', 'sCT']),
        EnsureChannelFirstd(keys=['CT', 'sCT'])
    ]
    
    # Add augmentation transforms for training
    if train:
        base_transforms.extend([
            RandRotate90d(
                keys=['CT', 'sCT'],
                prob=0.5,
                max_k=3,  # Random rotation of 0, 90, 180, or 270 degrees
                spatial_axes=[1, 2]  # Rotate in the 2D plane
            ),
            RandFlipd(
                keys=['CT', 'sCT'],
                prob=0.5,
                spatial_axis=1  # Horizontal flip
            ),
            RandFlipd(
                keys=['CT', 'sCT'],
                prob=0.5,
                spatial_axis=0  # Vertical flip
            )
        ])
    
    # Create transform pipeline
    transforms = Compose(base_transforms)
    
    # Create dataset
    dataset = CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def get_train_val_loaders(
    data_div_json: str,
    output_dir: str,
    cache_rate: float = 1.0,
    num_workers: int = 4,
    batch_size: int = 1,
    shuffle: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Get both training and validation dataloaders
    """
    train_loader = prepare_dataset(
        data_div_json=data_div_json,
        output_dir=output_dir,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        train=True
    )
    
    val_loader = prepare_dataset(
        data_div_json=data_div_json,
        output_dir=output_dir,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        train=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    data_div_json = "path/to/your/data_division.json"
    output_dir = "path/to/preprocessed/data"
    
    # Get dataloaders
    train_loader, val_loader = get_train_val_loaders(
        data_div_json=data_div_json,
        output_dir=output_dir,
        cache_rate=1.0,
        num_workers=4,
        batch_size=8
    )
    
    # Example of iterating through the dataloader
    for batch in train_loader:
        ct = batch['CT']  # Shape: (batch_size, 3, 256, 256)
        sct = batch['sCT']  # Shape: (batch_size, 3, 256, 256)
        print(f"Batch shapes - CT: {ct.shape}, sCT: {sct.shape}")
        break  # Just print first batch 