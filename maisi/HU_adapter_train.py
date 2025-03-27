import os
import torch
import numpy as np
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
from monai.losses import L1Loss
from monai.metrics import MAEMetric
from monai.utils import set_determinism

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Import case lists
from maisi.HU_adapter_UNet import train_case_name_list, test_case_name_list

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create data dictionary for training
train_files = []
for case_name in train_case_name_list:
    ct_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"
    sct_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_TS_MAISI.nii.gz"
    if os.path.exists(ct_path) and os.path.exists(sct_path):
        train_files.append({
            "ct": ct_path,
            "sct": sct_path
        })

# Create transforms - Correctly map HU range (-1024 to 1976) to (0 to 1)
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
    # Use RandSpatialCropd instead of RandCropByPosNegLabeld for synthesis task
    RandSpatialCropd(
        keys=["ct", "sct"],
        roi_size=(96, 96, 96),
        random_size=False,
    ),
    RandRotate90d(keys=["ct", "sct"], prob=0.2, spatial_axes=(0, 1)),
    ToTensord(keys=["ct", "sct"]),
])

# Create dataset and dataloader
train_dataset = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=4,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
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
# L1 loss is often better for image synthesis tasks
loss_function = L1Loss()
# Using AdamW instead of Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create evaluator metrics for synthesis
mae_metric = MAEMetric()

# Training loop
num_epochs = 300
val_interval = 5
best_metric = float('inf')  # Lower is better for MAE
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# Create directory for saving models
os.makedirs("HU_adapter_UNet", exist_ok=True)

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
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join("HU_adapter_UNet", f"model_epoch_{epoch+1}.pth"))

print("Training completed!") 