import os
import json
import torch
import numpy as np
import argparse
import sys
from datetime import datetime
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandSpatialCropd,
    RandRotate90d,
    ToTensord,
    Spacingd,
)
from monai.data import CacheDataset, DataLoader
from torch import nn
from monai.metrics import MAEMetric
from monai.utils import set_determinism

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train UNet for a specific fold and GPU')
parser.add_argument('--fold', type=int, required=True, help='Fold number (1-4)')
parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
args = parser.parse_args()

# Set up logging
log_dir = os.path.join("HU_adapter_UNet", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"fold{args.fold}_gpu{args.gpu}_detailed.log")
log_f = open(log_file, 'w')

# Define a custom print function to log all output
original_print = print
def custom_print(*args, **kwargs):
    output = " ".join(map(str, args))
    original_print(output, **kwargs)
    log_f.write(output + "\n")
    log_f.flush()

# Replace the print function with our custom one
print = custom_print

print(f"Starting training for fold {args.fold} on GPU {args.gpu} at {datetime.now()}")
print(f"Log file: {log_file}")

# Set deterministic training for reproducibility
set_determinism(seed=args.fold)  # Use different seed for each fold

# Set device based on argument
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (GPU {args.gpu})")

# Import path helper functions
from maisi.HU_adapter_UNet import get_ct_path, get_sct_path

# Load fold data
folds_path = os.path.join("HU_adapter_UNet", "folds.json")
if not os.path.exists(folds_path):
    raise FileNotFoundError(f"Folds file not found at {folds_path}. Run HU_adapter_create_folds.py first.")

with open(folds_path, 'r') as f:
    folds = json.load(f)

fold_key = f"fold_{args.fold}"
if fold_key not in folds:
    raise ValueError(f"Invalid fold number: {args.fold}. Available folds: {list(folds.keys())}")

train_cases = folds[fold_key]["train"]
val_cases = folds[fold_key]["val"]

print(f"Running training for {fold_key}")
print(f"Training cases: {len(train_cases)}")
print(f"Validation cases: {len(val_cases)}")

# Create data dictionaries for training and validation
train_files = []
for case_name in train_cases:
    ct_path = get_ct_path(case_name)
    sct_path = get_sct_path(case_name)
    if os.path.exists(ct_path) and os.path.exists(sct_path):
        train_files.append({
            "ct": ct_path,
            "sct": sct_path
        })
    else:
        print(f"Warning: Files for case {case_name} not found. CT exists: {os.path.exists(ct_path)}, SCT exists: {os.path.exists(sct_path)}")

val_files = []
for case_name in val_cases:
    ct_path = get_ct_path(case_name)
    sct_path = get_sct_path(case_name)
    if os.path.exists(ct_path) and os.path.exists(sct_path):
        val_files.append({
            "ct": ct_path,
            "sct": sct_path
        })
    else:
        print(f"Warning: Files for case {case_name} not found. CT exists: {os.path.exists(ct_path)}, SCT exists: {os.path.exists(sct_path)}")

print(f"Found {len(train_files)} valid training files and {len(val_files)} valid validation files")

if len(train_files) == 0 or len(val_files) == 0:
    print("Error: No valid training or validation files found. Exiting.")
    sys.exit(1)

# Create transforms for training and validation
train_transforms = Compose([
    LoadImaged(keys=["ct", "sct"]),
    EnsureChannelFirstd(keys=["ct", "sct"]),
    Spacingd(
        keys=["ct", "sct"],
        pixdim=(1.5, 1.5, 1.5),
        mode=("bilinear", "bilinear"),
    ),
    # Handle the HU range properly for synthesis task
    ScaleIntensityd(
        keys=["ct", "sct"],
        minv=-1024.0,
        maxv=1976.0,
        a_min=0.0,
        a_max=1.0,
        b_min=0.0,
        b_max=1.0,
    ),
    # Use RandSpatialCropd for synthesis task
    RandSpatialCropd(
        keys=["ct", "sct"],
        roi_size=(96, 96, 96),
        random_size=False,
    ),
    RandRotate90d(keys=["ct", "sct"], prob=0.2, spatial_axes=(0, 1)),
    ToTensord(keys=["ct", "sct"]),
])

val_transforms = Compose([
    LoadImaged(keys=["ct", "sct"]),
    EnsureChannelFirstd(keys=["ct", "sct"]),
    Spacingd(
        keys=["ct", "sct"],
        pixdim=(1.5, 1.5, 1.5),
        mode=("bilinear", "bilinear"),
    ),
    # Handle the HU range properly for synthesis task
    ScaleIntensityd(
        keys=["ct", "sct"],
        minv=-1024.0,
        maxv=1976.0,
        a_min=0.0,
        a_max=1.0,
        b_min=0.0,
        b_max=1.0,
    ),
    ToTensord(keys=["ct", "sct"]),
])

# Create datasets and dataloaders with reduced cache
print(f"Creating datasets with 0.25 cache rate")
train_dataset = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=0.25,  # Cache only 25% of data
    num_workers=4,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

val_dataset = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=0.25,  # Cache only 25% of data
    num_workers=4,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,  # Use batch size 1 for validation to handle different image sizes
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

# Create model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
model = model.to(device)

# Loss function and optimizer for synthesis task
loss_function = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create evaluator metrics for synthesis
mae_metric = MAEMetric()

# Create output directory for this fold
fold_dir = os.path.join("HU_adapter_UNet", f"fold_{args.fold}")
os.makedirs(fold_dir, exist_ok=True)

# Define HU scaler - to convert normalized values back to HU for metric reporting
def scale_to_hu(normalized_values):
    return normalized_values * (1976.0 - (-1024.0)) + (-1024.0)

# Training loop
num_epochs = 300
val_interval = 5
best_metric = float('inf')  # Lower is better for MAE
best_metric_epoch = -1
epoch_loss_values = []
val_metric_values = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, targets = batch_data["ct"].to(device), batch_data["sct"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Multiply loss by 3000 as requested
        loss = loss_function(outputs, targets) * 3000.0
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        if step % 10 == 0:
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_mae = 0
            val_step = 0
            for val_data in val_loader:
                val_step += 1
                val_inputs, val_targets = val_data["ct"].to(device), val_data["sct"].to(device)
                val_outputs = model(val_inputs)
                
                # Calculate MAE
                mae_metric(y_pred=val_outputs, y=val_targets)
                
            # Aggregate MAE metric and convert to HU
            mean_mae = mae_metric.aggregate().item()
            mean_mae_hu = mean_mae * (1976.0 - (-1024.0))  # Convert normalized MAE to HU scale
            mae_metric.reset()
            
            val_metric_values.append(mean_mae_hu)
            
            if mean_mae_hu < best_metric:
                best_metric = mean_mae_hu
                best_metric_epoch = epoch + 1
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'mae_hu': mean_mae_hu,
                }, os.path.join(fold_dir, "best_model.pth"))
                print(f"Saved new best model with MAE: {mean_mae_hu:.4f} HU")
            
            print(f"Validation MAE: {mean_mae_hu:.4f} HU, Best MAE: {best_metric:.4f} HU at epoch {best_metric_epoch}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(fold_dir, f"model_epoch_{epoch+1}.pth"))
    
    # Save training and validation curves
    np.save(os.path.join(fold_dir, "epoch_loss.npy"), np.array(epoch_loss_values))
    np.save(os.path.join(fold_dir, "val_mae_hu.npy"), np.array(val_metric_values))

print(f"Training completed for fold {args.fold}!")
print(f"Best validation MAE: {best_metric:.4f} HU at epoch {best_metric_epoch}")

# Close log file
log_f.close()
# Restore original print
print = original_print 