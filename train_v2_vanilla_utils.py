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




def train_or_eval_or_test_the_batch_cond(
        batch, 
        batch_size, 
        stage, model, 
        optimizer=None, 
        device=None
    ):

    pet = batch["PET"] # 1, z, 256, 256
    ct = batch["CT"] # 1, z, 256, 256
    body = batch["BODY"]
    body = body > 0
    
    # Ensure first dimension (z) is multiple of 16
    required_multiple = 16
    pad_z = (required_multiple - pet.shape[1] % required_multiple) % required_multiple
    if pad_z > 0:
        pet = torch.nn.functional.pad(pet, (0, 0, 0, 0, 0, pad_z, 0, 0), mode='constant', value=0)
        ct = torch.nn.functional.pad(ct, (0, 0, 0, 0, 0, pad_z, 0, 0), mode='constant', value=0)
        body = torch.nn.functional.pad(body, (0, 0, 0, 0, 0, pad_z, 0, 0), mode='constant', value=0)

    len_z = pet.shape[1]
    batch_per_eval = get_param("train_param")["batch_per_eval"]
    num_frames = get_param("num_frames")
    root_dir = get_param("root")

    case_loss_first = 0.0
    case_loss_second = 0.0
    case_loss_third = 0.0

    # First dimension (z-axis)
    indices_list_first = [i for i in range(1, ct.shape[1]-1)]
    random.shuffle(indices_list_first)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, ct.shape[2], ct.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[2], ct.shape[3]))
    
    # Process first dimension
    for index in indices_list_first:
        # Fill PET conditioning
        batch_x[batch_size_count, 0, :, :] = pet[:, index-1, :, :]
        batch_x[batch_size_count, 1, :, :] = pet[:, index, :, :]
        batch_x[batch_size_count, 2, :, :] = pet[:, index+1, :, :]

        # Fill CT target
        batch_y[batch_size_count, 0, :, :] = ct[:, index-1, :, :]
        batch_y[batch_size_count, 1, :, :] = ct[:, index, :, :]
        batch_y[batch_size_count, 2, :, :] = ct[:, index+1, :, :]

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list_first[-1]:
            continue
        else:
            case_loss_first += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0

    case_loss_first = case_loss_first / (len(indices_list_first) // batch_size + 1)

    # Second dimension (y-axis)
    indices_list_second = [i for i in range(1, ct.shape[2]-1)]
    random.shuffle(indices_list_second)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[3]))
    
    for index in indices_list_second:
        # Get slices and permute to correct orientation
        pet_slices = pet[:, :, index-1:index+2, :].squeeze(0)  # z, 3, x
        ct_slices = ct[:, :, index-1:index+2, :].squeeze(0)    # z, 3, x
        
        # Permute to get correct orientation
        pet_slices = pet_slices.permute(1, 0, 2)  # 3, z, x
        ct_slices = ct_slices.permute(1, 0, 2)    # 3, z, x
        
        batch_x[batch_size_count] = pet_slices
        batch_y[batch_size_count] = ct_slices
        
        batch_size_count += 1
        
        if batch_size_count < batch_size and index != indices_list_second[-1]:
            continue
        else:
            # print(f"batch_x shape {batch_x.shape}, batch_y shape {batch_y.shape}")
            case_loss_second += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0
            
    case_loss_second = case_loss_second / (len(indices_list_second) // batch_size + 1)

    # Third dimension (x-axis)
    indices_list_third = [i for i in range(1, ct.shape[3]-1)]
    random.shuffle(indices_list_third)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[2]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[2]))
    
    for index in indices_list_third:
        # Get slices and permute to correct orientation
        pet_slices = pet[:, :, :, index-1:index+2].squeeze(0)  # z, y, 3
        ct_slices = ct[:, :, :, index-1:index+2].squeeze(0)    # z, y, 3
        
        # Permute to get correct orientation
        pet_slices = pet_slices.permute(2, 0, 1)  # 3, z, y
        ct_slices = ct_slices.permute(2, 0, 1)    # 3, z, y
        
        batch_x[batch_size_count] = pet_slices
        batch_y[batch_size_count] = ct_slices
        
        batch_size_count += 1
        
        if batch_size_count < batch_size and index != indices_list_third[-1]:
            continue
        else:
            case_loss_third += process_batch(batch_x, batch_y, stage, model, optimizer, device)
            batch_size_count = 0
            
    case_loss_third = case_loss_third / (len(indices_list_third) // batch_size + 1)

    return case_loss_first, case_loss_second, case_loss_third

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
    
    cv = get_param("cv")
    root = get_param("root")
    
    # cv = 0, 1, 2, 3, 4
    cv_test = cv
    cv_val = (cv+1)%5
    cv_train = [(cv+2)%5, (cv+3)%5, (cv+4)%5]

    train_list = data_div[f"cv{cv_train[0]}"] + data_div[f"cv{cv_train[1]}"] + data_div[f"cv{cv_train[2]}"]
    val_list = data_div[f"cv{cv_val}"]
    test_list = data_div[f"cv{cv_test}"]

    set_param("train_list", train_list)
    set_param("val_list", val_list)
    set_param("test_list", test_list)

    print(f"train_list:", train_list)
    print(f"val_list:", val_list)
    print(f"test_list:", test_list)

    # train_list: ['E4058', 'E4217', 'E4166', 'E4165', 'E4092', 'E4163', 'E4193', 'E4105', 'E4125', 'E4198', 'E4157', 'E4139', 'E4207', 'E4106', 'E4068', 'E4241', 'E4219', 'E4078', 'E4147', 'E4138', 'E4096', 'E4152', 'E4073', 'E4181', 'E4187', 'E4099', 'E4077', 'E4134', 'E4091', 'E4144', 'E4114', 'E4130', 'E4103', 'E4239', 'E4183', 'E4208', 'E4120', 'E4220', 'E4137', 'E4069', 'E4189', 'E4182']
    # val_list: ['E4216', 'E4081', 'E4118', 'E4074', 'E4079', 'E4094', 'E4115', 'E4237', 'E4084', 'E4061', 'E4055', 'E4098', 'E4232']
    # test_list: ['E4128', 'E4172', 'E4238', 'E4158', 'E4129', 'E4155', 'E4143', 'E4197', 'E4185', 'E4131', 'E4162', 'E4066', 'E4124']

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
            "filename": hashname,
        })

    for hashname in val_list:
        val_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
            "filename": hashname,
        })

    for hashname in test_list:
        test_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
            "filename": hashname,
        })

    # save the data division file
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

    input_modality = ["PET", "CT", "BODY"]  

    # set the data transform
    if invlove_train:
        train_transforms = Compose(
            [
                LoadImaged(keys=input_modality, image_only=True),
                EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
                EnsureTyped(keys=input_modality),
                # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["x"], 
                #     roi_size=(img_size, img_size, in_channel), 
                #     random_size=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["y"],
                #     roi_size=(img_size, img_size, out_channel),
                #     random_size=False),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["x"],
                #     channel_dim=-1),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["y"],
                #     channel_dim="none" if out_channel == 1 else -1),

            ]
        )

    if invlove_val:
        val_transforms = Compose(
            [
                LoadImaged(keys=input_modality, image_only=True),
                EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
                EnsureTyped(keys=input_modality),
                # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["x"], 
                #     roi_size=(img_size, img_size, in_channel), 
                #     random_size=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["y"],
                #     roi_size=(img_size, img_size, out_channel),
                #     random_size=False),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["x"],
                #     channel_dim=-1),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["y"],
                #     channel_dim="none" if out_channel == 1 else -1),
            ]
        )
    if invlove_test:
        test_transforms = Compose(
            [
                LoadImaged(keys=input_modality, image_only=True),
                EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
                EnsureTyped(keys=input_modality),
                # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["x"], 
                #     roi_size=(img_size, img_size, in_channel), 
                #     random_size=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["y"],
                #     roi_size=(img_size, img_size, out_channel),
                #     random_size=False),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["x"],
                #     channel_dim=-1),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["y"],
                #     channel_dim="none" if out_channel == 1 else -1),
            ]
        )

    
    if invlove_train:
        train_ds = CacheDataset(
            data=train_path_list,
            transform=train_transforms,
            # cache_num=num_train_files,
            cache_rate=get_param("data_param")["dataset"]["train"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["train"]["num_workers"],
        )

    if invlove_val:
        val_ds = CacheDataset(
            data=val_path_list,
            transform=val_transforms, 
            # cache_num=num_val_files,
            cache_rate=get_param("data_param")["dataset"]["val"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["val"]["num_workers"],
        )

    if invlove_test:
        test_ds = CacheDataset(
            data=test_path_list,
            transform=test_transforms,
            # cache_num=num_test_files,
            cache_rate=get_param("data_param")["dataset"]["test"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["test"]["num_workers"],
        )

    if invlove_train:
        train_loader = DataLoader(
            train_ds, 
            batch_size=1,
            shuffle=True,
            num_workers=get_param("data_param")["dataloader"]["train"]["num_workers"],
        )

    if invlove_val:
        val_loader = DataLoader(
            val_ds, 
            batch_size=1,
            shuffle=False,
            num_workers=get_param("data_param")["dataloader"]["val"]["num_workers"],
        )

    if invlove_test:
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=get_param("data_param")["dataloader"]["test"]["num_workers"],
        )

    if not invlove_train:
        train_loader = None
    
    if not invlove_val:
        val_loader = None

    if not invlove_test:
        test_loader = None
    
    return train_loader, val_loader, test_loader

# from torchvision.models import inception_v3
# from scipy.linalg import sqrtm
# from torchvision import transforms
# import torch.nn.functional as F

# def load_inception_model(weights_path):
#     """
#     Load InceptionV3 model from a given path. If the model weights are not found at the path,
#     download the pretrained model, save the weights to the path, and return the model.
    
#     Args:
#         weights_path (str): Path to the file where pretrained weights are stored or will be saved.
    
#     Returns:
#         torch.nn.Module: The InceptionV3 model loaded with weights.
#     """
#     model = inception_v3(pretrained=False, transform_input=False)  # Initialize the model

#     if os.path.exists(weights_path):
#         # Load weights locally if the file exists
#         print(f"Loading pretrained weights from {weights_path}...")
#         model.load_state_dict(torch.load(weights_path))
#     else:
#         # Download pretrained weights, save them locally, and load them
#         print(f"Pretrained weights not found at {weights_path}. Downloading and saving locally...")
#         model = inception_v3(pretrained=True, transform_input=False)
#         torch.save(model.state_dict(), weights_path)
    
#     model.eval()  # Set model to evaluation mode
#     return model

# def get_features(images, model):
#     with torch.no_grad():
#         features = model(images).detach()
#     return features

# def compute_FID(real_y, recon_y, inception):
#     # Load pre-trained InceptionV3 model
#     # inception = inception_v3(pretrained=True, transform_input=False)
#     # inception.eval()  # Set the model to evaluation mode
    
#     # Step 1: Rescale to [0, 1]
#     real_y = (real_y + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
#     recon_y = (recon_y + 1) / 2.0 # Rescale from [-1, 1] to [0, 1]
    
#     # Step 3: Resize to 299x299
#     real_y = F.interpolate(real_y, size=(299, 299), mode='bilinear', align_corners=False)
#     recon_y = F.interpolate(recon_y, size=(299, 299), mode='bilinear', align_corners=False)
    
#     # Step 4: Normalize using ImageNet statistics
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
#     real_y = torch.stack([normalize(img) for img in real_y])
#     recon_y = torch.stack([normalize(img) for img in recon_y])

#     # Extract features for both real and reconstructed images
#     real_features = get_features(real_y, inception)
#     recon_features = get_features(recon_y, inception)

#     real_features_np = real_features.cpu().numpy()
#     recon_features_np = recon_features.cpu().numpy()

#     # Compute mean and covariance
#     real_mu = np.mean(real_features_np, axis=0)
#     real_sigma = np.cov(real_features_np, rowvar=False)

#     recon_mu = np.mean(recon_features_np, axis=0)
#     recon_sigma = np.cov(recon_features_np, rowvar=False)

#     # Compute the squared difference between means
#     mean_diff = np.sum((real_mu - recon_mu) ** 2)

#     # Compute the square root of the product of covariance matrices
#     cov_sqrt = sqrtm(real_sigma @ recon_sigma)

#     # Handle numerical errors from sqrtm (e.g., small imaginary components)
#     if np.iscomplexobj(cov_sqrt):
#         cov_sqrt = cov_sqrt.real

#     # Compute the FID score
#     fid = mean_diff + np.trace(real_sigma + recon_sigma - 2 * cov_sqrt)

#     return fid