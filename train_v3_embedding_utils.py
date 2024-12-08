import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import json
import random

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
    EnsureTyped,
)
import torch.nn.functional as F

import time
from monai.data import CacheDataset, DataLoader
from global_config import global_config, set_param, get_param
# take the psnr as the metric from skimage
from skimage.metrics import peak_signal_noise_ratio as psnr

def printlog(message):
    log_txt_path = get_param("log_txt_path")
    # attach the current time as YYYY-MM-DD HH:MM:SS 
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message = f"{current_time} {message}"
    print(message)
    with open(log_txt_path, "a") as f:
        f.write(message)
        f.write("\n")


@torch.inference_mode()
def test_diffusion_model_and_save_slices(data_loader, model, device, output_dir, batch_size=8):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    print("Starting testing...")
    num_case = len(data_loader)

    for idx_case, batch in enumerate(data_loader):
        printlog(f"Processing case {idx_case + 1}/{len(data_loader)}")

        filenames = batch["filename"]
        pet = batch["PET"].to(device)  # Shape: (1, z, 256, 256)
        ct = batch["CT"].to(device)  # Ground truth CT, Shape: (1, z, 256, 256)
        len_z = pet.shape[1]  # Number of slices along the z-axis

        # Process slices in batches
        for z_start in range(1, len_z - 1, batch_size):
            # Calculate actual batch size for this iteration
            current_batch_size = min(batch_size, len_z - 1 - z_start)
            
            # Create batch for model input
            cond = torch.zeros((current_batch_size, 3, pet.shape[2], pet.shape[3])).to(device)
            
            # Fill the conditioning tensor for each slice in the batch
            for i in range(current_batch_size):
                z = z_start + i
                cond[i, 0, :, :] = pet[:, z - 1, :, :]
                cond[i, 1, :, :] = pet[:, z, :, :]
                cond[i, 2, :, :] = pet[:, z + 1, :, :]

            # Generate predictions for the batch
            pred_slices = model.sample(batch_size=current_batch_size, cond=cond)  # Shape: (batch_size, 3, h, w)

            # Process and save each slice in the batch
            for i in range(current_batch_size):
                z = z_start + i
                
                # Select middle slice prediction
                pred_slice = pred_slices[i]  # Shape: (3, h, w)
                pred_slice_clipped = torch.clamp(pred_slice[1, :, :], min=-1, max=1)
                pred_slice_normalized = (pred_slice_clipped + 1) / 2.0
                ct_slice_normalized = (ct[:, z, :, :] + 1) / 2.0
                pet_slice = pet[:, z, :, :].cpu().numpy()

                # Compute MAE loss with a factor of 4000
                slice_mae = F.l1_loss(pred_slice_normalized, ct_slice_normalized, reduction="mean") * 4000
                printlog(f"Case {idx_case + 1}/{num_case}, Slice {z}/{len_z}: MAE = {slice_mae.item():.6f}")

                # Save data for this slice
                save_data = {
                    "PET": pet_slice,
                    "CT": ct_slice_normalized.cpu().numpy(),
                    "Pred_CT": pred_slice_normalized.cpu().numpy(),
                    "MAE": slice_mae.item()
                }
                save_path = os.path.join(output_dir, f"{filenames[0]}_case_{idx_case + 1}_slice_{z}.npz")
                np.savez_compressed(save_path, **save_data)

                printlog(f"Saved slice {z} for case {idx_case + 1} to {save_path} at MAE {slice_mae.item()}")

    printlog("Testing and saving completed.")



def train_or_eval_or_test_the_batch_cond(
        batch, 
        batch_size, 
        stage, model, 
        optimizer=None, 
        device=None
    ):
    # Extract data for all three views
    x_axial = batch["x_axial"].squeeze(0).to(device)  # shape: (len_z, 3, h, w)
    y_axial = batch["y_axial"].squeeze(0).to(device)
    x_coronal = batch["x_coronal"].squeeze(0).to(device)  # shape: (len_y, 3, h, w)
    y_coronal = batch["y_coronal"].squeeze(0).to(device)
    x_sagittal = batch["x_sagittal"].squeeze(0).to(device)  # shape: (len_x, 3, h, w)
    y_sagittal = batch["y_sagittal"].squeeze(0).to(device)

    # Ensure first dimension is multiple of 16 for each view
    required_multiple = 16
    
    # Pad axial view
    pad_z = (required_multiple - x_axial.shape[0] % required_multiple) % required_multiple
    if pad_z > 0:
        x_axial = torch.nn.functional.pad(x_axial, (0, 0, 0, 0, 0, 0, 0, pad_z), mode='constant', value=0)
        y_axial = torch.nn.functional.pad(y_axial, (0, 0, 0, 0, 0, 0, 0, pad_z), mode='constant', value=0)
    
    # Pad coronal view
    pad_y = (required_multiple - x_coronal.shape[0] % required_multiple) % required_multiple
    if pad_y > 0:
        x_coronal = torch.nn.functional.pad(x_coronal, (0, 0, 0, 0, 0, 0, 0, pad_y), mode='constant', value=0)
        y_coronal = torch.nn.functional.pad(y_coronal, (0, 0, 0, 0, 0, 0, 0, pad_y), mode='constant', value=0)
    
    # Pad sagittal view
    pad_x = (required_multiple - x_sagittal.shape[0] % required_multiple) % required_multiple
    if pad_x > 0:
        x_sagittal = torch.nn.functional.pad(x_sagittal, (0, 0, 0, 0, 0, 0, 0, pad_x), mode='constant', value=0)
        y_sagittal = torch.nn.functional.pad(y_sagittal, (0, 0, 0, 0, 0, 0, 0, pad_x), mode='constant', value=0)

    case_loss_axial = 0.0
    case_loss_coronal = 0.0
    case_loss_sagittal = 0.0

    # Process axial view
    indices_list = [i for i in range(1, x_axial.shape[0]-1)]  # range over len_z
    random.shuffle(indices_list)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, x_axial.shape[2], x_axial.shape[3])).to(device)
    batch_y = torch.zeros((batch_size, 3, x_axial.shape[2], x_axial.shape[3])).to(device)
    
    for index in indices_list:
        # Fill PET conditioning - x_axial[index] shape is (3, h, w)
        batch_x[batch_size_count] = x_axial[index]  # Current slice
        batch_y[batch_size_count] = y_axial[index]  # Current slice

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list[-1]:
            continue
        else:
            if stage == "train":
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    batch_x = torch.rot90(batch_x, k=k, dims=[-2, -1])
                    batch_y = torch.rot90(batch_y, k=k, dims=[-2, -1])
                
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-1])
                    batch_y = torch.flip(batch_y, dims=[-1])
                
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-2])
                    batch_y = torch.flip(batch_y, dims=[-2])

            case_loss_axial += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0

    case_loss_axial = case_loss_axial / (len(indices_list) // batch_size + 1)

    # Process coronal view
    indices_list = [i for i in range(1, x_coronal.shape[0]-1)]  # range over len_y
    random.shuffle(indices_list)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, x_coronal.shape[2], x_coronal.shape[3])).to(device)
    batch_y = torch.zeros((batch_size, 3, x_coronal.shape[2], x_coronal.shape[3])).to(device)
    
    for index in indices_list:
        batch_x[batch_size_count] = x_coronal[index]
        batch_y[batch_size_count] = y_coronal[index]

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list[-1]:
            continue
        else:
            if stage == "train":
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    batch_x = torch.rot90(batch_x, k=k, dims=[-2, -1])
                    batch_y = torch.rot90(batch_y, k=k, dims=[-2, -1])
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-1])
                    batch_y = torch.flip(batch_y, dims=[-1])
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-2])
                    batch_y = torch.flip(batch_y, dims=[-2])

            case_loss_coronal += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0
            
    case_loss_coronal = case_loss_coronal / (len(indices_list) // batch_size + 1)

    # Process sagittal view
    indices_list = [i for i in range(1, x_sagittal.shape[0]-1)]  # range over len_x
    random.shuffle(indices_list)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, x_sagittal.shape[2], x_sagittal.shape[3])).to(device)
    batch_y = torch.zeros((batch_size, 3, x_sagittal.shape[2], x_sagittal.shape[3])).to(device)
    
    for index in indices_list:
        batch_x[batch_size_count] = x_sagittal[index]
        batch_y[batch_size_count] = y_sagittal[index]

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list[-1]:
            continue
        else:
            if stage == "train":
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    batch_x = torch.rot90(batch_x, k=k, dims=[-2, -1])
                    batch_y = torch.rot90(batch_y, k=k, dims=[-2, -1])
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-1])
                    batch_y = torch.flip(batch_y, dims=[-1])
                if random.random() < 0.5:
                    batch_x = torch.flip(batch_x, dims=[-2])
                    batch_y = torch.flip(batch_y, dims=[-2])

            case_loss_sagittal += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0
            
    case_loss_sagittal = case_loss_sagittal / (len(indices_list) // batch_size + 1)

    return case_loss_axial, case_loss_coronal, case_loss_sagittal

def process_batch(batch_x, batch_y, stage, model, optimizer, device):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    if stage == "train":
        optimizer.zero_grad()
        loss = model(img=batch_y, cond=batch_x)
        loss.backward()
        optimizer.step()
        return loss.item()
    elif stage == "eval" or stage == "test":
        with torch.no_grad():
            loss = model(img=batch_y, cond=batch_x)
            return loss.item()

def prepare_dataset(data_div, invlove_train=False, invlove_val=False, invlove_test=False):
    # Get configuration parameters
    cv = get_param("cv")
    root = get_param("root")
    
    # Calculate cross-validation splits
    cv_splits = _calculate_cv_splits(cv, data_div)
    train_list, val_list, test_list = cv_splits
    
    # Save splits to global parameters
    _save_splits_to_params(train_list, val_list, test_list)
    
    # Construct path lists for each split
    path_lists = _construct_path_lists(train_list, val_list, test_list)
    train_path_list, val_path_list, test_path_list = path_lists
    
    # Save data division to JSON
    _save_data_division(root, train_path_list, val_path_list, test_path_list)
    
    # Create transforms and datasets
    input_modality = ["axial", "coronal", "sagittal"]
    transforms = _create_transforms(input_modality, invlove_train, invlove_val, invlove_test)
    
    # Create dataloaders
    loaders = _create_dataloaders(
        transforms,
        path_lists,
        invlove_train,
        invlove_val,
        invlove_test
    )
    
    return loaders

def _calculate_cv_splits(cv, data_div):
    """Calculate cross-validation splits."""
    cv_test = cv
    cv_val = (cv+1)%5
    cv_train = [(cv+2)%5, (cv+3)%5, (cv+4)%5]

    train_list = []
    for i in cv_train:
        train_list.extend(data_div[f"cv{i}"])
    val_list = data_div[f"cv{cv_val}"]
    test_list = data_div[f"cv{cv_test}"]
    
    return train_list, val_list, test_list

def _construct_path_lists(train_list, val_list, test_list):
    """Construct file path lists for each split."""
    def _create_paths(hashname):
        return {
            "axial": f"James_data_v3/embeddings/{hashname}_axial_embedding_norm.npz",
            "coronal": f"James_data_v3/embeddings/{hashname}_coronal_embedding_norm.npz",
            "sagittal": f"James_data_v3/embeddings/{hashname}_sagittal_embedding_norm.npz",
            "filename": hashname,
        }
    
    train_path_list = [_create_paths(h) for h in train_list]
    val_path_list = [_create_paths(h) for h in val_list]
    test_path_list = [_create_paths(h) for h in test_list]
    
    return train_path_list, val_path_list, test_path_list

def _save_splits_to_params(train_list, val_list, test_list):
    """Save splits to global parameters."""
    set_param("train_list", train_list)
    set_param("val_list", val_list)
    set_param("test_list", test_list)

    print(f"train_list:", train_list)
    print(f"val_list:", val_list)
    print(f"test_list:", test_list)

def _save_data_division(root, train_path_list, val_path_list, test_path_list):
    """Save data division to JSON."""
    data_division_file = os.path.join(root, "data_division.json")
    data_division_dict = {
        "train": train_path_list,
        "val": val_path_list,
        "test": test_path_list,
    }
    for key in data_division_dict.keys():
        print(key)
        for key2 in data_division_dict[key]:
            print(key2)

    with open(data_division_file, "w") as f:
        json.dump(data_division_dict, f, indent=4)

def _create_transforms(input_modality, invlove_train, invlove_val, invlove_test):
    """Create transforms for each split."""
    class LoadNpzEmbedding(monai.transforms.Transform):
        def __call__(self, data):
            # Load each orientation's NPZ file and split into x_ and y_ data
            for orientation in ["axial", "coronal", "sagittal"]:
                if orientation in data:
                    npz_data = np.load(data[orientation])
                    
                    # Create x_ and y_ keys for this orientation
                    x_key = f"x_{orientation}"
                    y_key = f"y_{orientation}"
                    
                    # Load PET embedding into x_ and CT embedding into y_
                    data[x_key] = npz_data["pet_embedding"]  # shape: (n_slices, 3, h, w)
                    data[y_key] = npz_data["ct_embedding"]  # shape: (n_slices, 3, h, w)
                    
                    # Verify shapes match
                    expected_shape = tuple(npz_data["shape"])
                    assert data[x_key].shape == expected_shape, f"Shape mismatch for {x_key}"
                    assert data[y_key].shape == expected_shape, f"Shape mismatch for {y_key}"
                    
            return data

    transforms_list = [
        LoadNpzEmbedding(),
        EnsureTyped(keys=["x_axial", "y_axial", "x_coronal", "y_coronal", "x_sagittal", "y_sagittal"]),
    ]

    transforms = Compose(transforms_list)

    if invlove_train:
        train_ds = CacheDataset(
            data=train_path_list,
            transform=transforms,
            cache_rate=get_param("data_param")["dataset"]["train"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["train"]["num_workers"],
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=1,  # Keep batch_size=1 as we handle batching in train_or_eval_or_test_the_batch_cond
            shuffle=True,
            num_workers=get_param("data_param")["dataloader"]["train"]["num_workers"],
        )
    else:
        train_loader = None

    if invlove_val:
        val_ds = CacheDataset(
            data=val_path_list,
            transform=transforms,
            cache_rate=get_param("data_param")["dataset"]["val"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["val"]["num_workers"],
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=1,
            shuffle=False,
            num_workers=get_param("data_param")["dataloader"]["val"]["num_workers"],
        )
    else:
        val_loader = None

    if invlove_test:
        test_ds = CacheDataset(
            data=test_path_list,
            transform=transforms,
            cache_rate=get_param("data_param")["dataset"]["test"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["test"]["num_workers"],
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=get_param("data_param")["dataloader"]["test"]["num_workers"],
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
