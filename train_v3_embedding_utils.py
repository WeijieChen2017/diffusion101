import os
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import monai
from monai.transforms import (
    Compose,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader
from global_config import set_param, get_param

def get_loss_function(loss_type):
    """
    Get the loss function based on the config setting.
    Args:
        loss_type: str, one of "L1", "L2", or "COS"
    Returns:
        loss_fn: callable, the loss function
    """
    if loss_type == "L2":
        return F.mse_loss
    elif loss_type == "L1":
        return F.l1_loss
    elif loss_type == "COS":
        def cosine_loss(input, target, reduction='none'):
            # Normalize the vectors
            input_normalized = F.normalize(input.flatten(2), dim=2)
            target_normalized = F.normalize(target.flatten(2), dim=2)
            # Compute cosine similarity (1 - cos_sim for loss)
            cos_sim = 1 - (input_normalized * target_normalized).sum(dim=2)
            if reduction == 'none':
                return cos_sim
            elif reduction == 'mean':
                return cos_sim.mean()
            else:
                raise ValueError(f"Unsupported reduction mode: {reduction}")
        return cosine_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


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
def test_diffusion_model_unit_sphere_and_save_slices(data_loader, model, device, output_dir, vq_weights, batch_size=8):
    """
    Args:
        vq_weights: torch.Tensor of shape (8192, 3), the VQ codebook
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize VQ weights to unit vectors
    vq_weights = torch.from_numpy(vq_weights).to(device)  # (8192, 3)
    vq_weights = F.normalize(vq_weights, p=2, dim=1)  # normalize each 1x3 vector

    def normalize_to_unit_sphere(x):
        """Normalize each 1x3 vector to unit length.
        Args:
            x: tensor of shape (slices, 3, h, w) or (batch, 3, h, w)
        Returns:
            normalized tensor of same shape
        """
        # Reshape to (...*h*w, 3)
        orig_shape = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
        # Normalize each vector to unit length
        x_norm = F.normalize(x_flat, p=2, dim=1)
        # Reshape back
        x_norm = x_norm.reshape(orig_shape[0], orig_shape[2], orig_shape[3], 3).permute(0, 3, 1, 2)
        return x_norm

    def find_nearest_embedding_cosine(pred_embedding):
        """
        Find the nearest VQ embedding using cosine similarity
        Args:
            pred_embedding: tensor of shape (3, h, w)
        Returns:
            quantized embedding of shape (3, h, w)
        """
        h, w = pred_embedding.shape[1:]
        flat_pred = pred_embedding.permute(1, 2, 0).reshape(-1, 3)  # (h*w, 3)
        
        # Normalize prediction vectors
        flat_pred = F.normalize(flat_pred, p=2, dim=1)  # normalize each 1x3 vector
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = torch.mm(flat_pred, vq_weights.t())  # (h*w, 8192)
        
        # Find most similar vectors
        nearest_indices = torch.argmax(similarities, dim=1)  # (h*w,)
        quantized = vq_weights[nearest_indices]  # (h*w, 3)
        quantized = quantized.reshape(h, w, 3).permute(2, 0, 1)  # (3, h, w)
        return quantized

    def compute_cosine_similarity(pred, target):
        """
        Compute mean cosine similarity between prediction and target
        Args:
            pred, target: tensors of shape (3, h, w)
        Returns:
            mean cosine similarity (scalar)
        """
        # Reshape to (h*w, 3)
        pred_flat = pred.permute(1, 2, 0).reshape(-1, 3)
        target_flat = target.permute(1, 2, 0).reshape(-1, 3)
        
        # Normalize vectors
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        target_norm = F.normalize(target_flat, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (h*w,)
        return cos_sim.mean()

    print("Starting testing...")
    num_case = len(data_loader)

    for idx_case, batch in enumerate(data_loader):
        printlog(f"Processing case {idx_case + 1}/{len(data_loader)}")

        # Extract and normalize data for all three views
        x_axial = normalize_to_unit_sphere(batch["x_axial"].squeeze(0).to(device))
        y_axial = normalize_to_unit_sphere(batch["y_axial"].squeeze(0).to(device))
        x_coronal = normalize_to_unit_sphere(batch["x_coronal"].squeeze(0).to(device))
        y_coronal = normalize_to_unit_sphere(batch["y_coronal"].squeeze(0).to(device))
        x_sagittal = normalize_to_unit_sphere(batch["x_sagittal"].squeeze(0).to(device))
        y_sagittal = normalize_to_unit_sphere(batch["y_sagittal"].squeeze(0).to(device))
        filename = batch["filename"][0]

        # Process each view
        for view_name, x, y in [
            ("axial", x_axial, y_axial),
            ("coronal", x_coronal, y_coronal),
            ("sagittal", x_sagittal, y_sagittal),
        ]:
            len_slices = x.shape[0]
            
            # Calculate padding
            required_multiple = 8
            pad_h = (required_multiple - x.shape[2] % required_multiple) % required_multiple
            pad_w = (required_multiple - x.shape[3] % required_multiple) % required_multiple
            
            # Store original dimensions before padding
            original_h, original_w = x.shape[2], x.shape[3]
            
            # Apply padding if needed
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
                y = F.pad(y, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
            # Process slices in batches
            for slice_start in range(0, len_slices, batch_size):
                current_batch_size = min(batch_size, len_slices - 1 - slice_start)
                
                batch_x = torch.zeros((current_batch_size, 3, x.shape[2], x.shape[3])).to(device)
                batch_y = torch.zeros((current_batch_size, 3, x.shape[2], x.shape[3])).to(device)
                
                for i in range(current_batch_size):
                    slice_idx = slice_start + i
                    batch_x[i] = x[slice_idx]
                    batch_y[i] = y[slice_idx]

                # Generate predictions for the batch
                pred_slices = model.sample(batch_size=current_batch_size, cond=batch_x)
                
                # Remove padding from predictions and inputs
                if pad_h > 0 or pad_w > 0:
                    pred_slices = pred_slices[:, :, :original_h, :original_w]
                    batch_x = batch_x[:, :, :original_h, :original_w]
                    batch_y = batch_y[:, :, :original_h, :original_w]

                # Process and save each slice in the batch
                for i in range(current_batch_size):
                    slice_idx = slice_start + i
                    
                    pred_slice = pred_slices[i]
                    gt_slice = batch_y[i]
                    cond_slice = batch_x[i]

                    # Find nearest VQ embedding using cosine similarity
                    vq_pred_slice = find_nearest_embedding_cosine(pred_slice)

                    # Compute cosine similarities
                    pred_sim = compute_cosine_similarity(pred_slice, gt_slice)
                    vq_sim = compute_cosine_similarity(vq_pred_slice, gt_slice)
                    
                    printlog(f"Case {idx_case + 1}/{num_case}, {view_name} Slice {slice_idx}/{len_slices}: "
                           f"Pred Similarity = {pred_sim.item():.6f}, VQ Similarity = {vq_sim.item():.6f}")

                    # Save data (all tensors are already unpadded)
                    save_data = {
                        "cond_embedding": cond_slice.cpu().numpy(),
                        "gt_embedding": gt_slice.cpu().numpy(),
                        "pred_embedding": pred_slice.cpu().numpy(),
                        "vq_pred_embedding": vq_pred_slice.cpu().numpy(),
                        "pred_similarity": pred_sim.item(),
                        "vq_similarity": vq_sim.item()
                    }
                    save_path = os.path.join(output_dir, f"{filename}_case_{idx_case + 1}_{view_name}_slice_{slice_idx}.npz")
                    np.savez_compressed(save_path, **save_data)

                    printlog(f"Saved {view_name} slice {slice_idx} for case {idx_case + 1} to {save_path}")

    printlog("Testing and saving completed.")

@torch.inference_mode()
def test_diffusion_model_and_save_slices(data_loader, model, device, output_dir, vq_weights, batch_size=8):
    """
    Args:
        vq_weights: torch.Tensor of shape (8192, 3), the VQ codebook
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    vq_weights = torch.from_numpy(vq_weights).to(device)  # (8192, 3)

    def find_nearest_embedding(pred_embedding):
        """
        Find the nearest VQ embedding for each pixel position
        Args:
            pred_embedding: tensor of shape (3, h, w)
        Returns:
            quantized embedding of shape (3, h, w)
        """
        h, w = pred_embedding.shape[1:]
        flat_pred = pred_embedding.permute(1, 2, 0).reshape(-1, 3)  # (h*w, 3)
        distances = torch.cdist(flat_pred, vq_weights)  # (h*w, 8192)
        nearest_indices = torch.argmin(distances, dim=1)  # (h*w,)
        quantized = vq_weights[nearest_indices]  # (h*w, 3)
        quantized = quantized.reshape(h, w, 3).permute(2, 0, 1)  # (3, h, w)
        return quantized

    print("Starting testing...")
    num_case = len(data_loader)

    for idx_case, batch in enumerate(data_loader):
        printlog(f"Processing case {idx_case + 1}/{len(data_loader)}")

        # Extract data for all three views
        x_axial = batch["x_axial"].squeeze(0).to(device)
        y_axial = batch["y_axial"].squeeze(0).to(device)
        x_coronal = batch["x_coronal"].squeeze(0).to(device)
        y_coronal = batch["y_coronal"].squeeze(0).to(device)
        x_sagittal = batch["x_sagittal"].squeeze(0).to(device)
        y_sagittal = batch["y_sagittal"].squeeze(0).to(device)
        filename = batch["filename"][0]

        # Print original shapes
        # printlog(f"Original shapes:")
        # printlog(f"x_axial: {x_axial.shape}")
        # printlog(f"x_coronal: {x_coronal.shape}")
        # printlog(f"x_sagittal: {x_sagittal.shape}")

        # Process each view
        for view_name, x, y in [
            ("axial", x_axial, y_axial),
            ("coronal", x_coronal, y_coronal),
            ("sagittal", x_sagittal, y_sagittal),
        ]:
            len_slices = x.shape[0]
            
            # Calculate padding
            required_multiple = 8
            pad_h = (required_multiple - x.shape[2] % required_multiple) % required_multiple
            pad_w = (required_multiple - x.shape[3] % required_multiple) % required_multiple
            
            # Store original dimensions before padding
            original_h, original_w = x.shape[2], x.shape[3]
            
            # Apply padding if needed
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
                y = F.pad(y, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                # printlog(f"{view_name} view dimensions:")
                # printlog(f"Original shape: ({original_h}, {original_w})")
                # printlog(f"Padded shape: {x.shape}")
                # printlog(f"Padding added: height={pad_h}, width={pad_w}")
            
            # Process slices in batches
            for slice_start in range(1, len_slices - 1, batch_size):
                current_batch_size = min(batch_size, len_slices - 1 - slice_start)
                
                batch_x = torch.zeros((current_batch_size, 3, x.shape[2], x.shape[3])).to(device)
                batch_y = torch.zeros((current_batch_size, 3, x.shape[2], x.shape[3])).to(device)
                
                for i in range(current_batch_size):
                    slice_idx = slice_start + i
                    batch_x[i] = x[slice_idx]
                    batch_y[i] = y[slice_idx]

                # Generate predictions for the batch
                pred_slices = model.sample(batch_size=current_batch_size, cond=batch_x)
                
                # Remove padding from predictions and inputs
                if pad_h > 0 or pad_w > 0:
                    pred_slices = pred_slices[:, :, :original_h, :original_w]
                    batch_x = batch_x[:, :, :original_h, :original_w]
                    batch_y = batch_y[:, :, :original_h, :original_w]

                # Process and save each slice in the batch
                for i in range(current_batch_size):
                    slice_idx = slice_start + i
                    
                    pred_slice = pred_slices[i]
                    gt_slice = batch_y[i]
                    cond_slice = batch_x[i]

                    # Find nearest VQ embedding
                    vq_pred_slice = find_nearest_embedding(pred_slice)

                    # Compute losses (now using unpadded tensors)
                    pred_loss = F.l1_loss(pred_slice, gt_slice, reduction="mean")
                    vq_loss = F.l1_loss(vq_pred_slice, gt_slice, reduction="mean")
                    
                    printlog(f"Case {idx_case + 1}/{num_case}, {view_name} Slice {slice_idx}/{len_slices}: "
                           f"Pred Loss = {pred_loss.item():.6f}, VQ Loss = {vq_loss.item():.6f}")

                    # Save data (all tensors are already unpadded)
                    save_data = {
                        "cond_embedding": cond_slice.cpu().numpy(),
                        "gt_embedding": gt_slice.cpu().numpy(),
                        "pred_embedding": pred_slice.cpu().numpy(),
                        "vq_pred_embedding": vq_pred_slice.cpu().numpy(),
                        "pred_loss": pred_loss.item(),
                        "vq_loss": vq_loss.item()
                    }
                    save_path = os.path.join(output_dir, f"{filename}_case_{idx_case + 1}_{view_name}_slice_{slice_idx}.npz")
                    np.savez_compressed(save_path, **save_data)

                    printlog(f"Saved {view_name} slice {slice_idx} for case {idx_case + 1} to {save_path}")

    printlog("Testing and saving completed.")



def train_or_eval_or_test_the_batch_cond(
        batch, 
        batch_size, 
        stage, model, 
        optimizer=None, 
        device=None,
        loss_fn=F.mse_loss
    ):
    # Extract data for all three views
    x_axial = batch["x_axial"].squeeze(0).to(device)  # shape: (len_z, 3, h, w)
    y_axial = batch["y_axial"].squeeze(0).to(device)
    x_coronal = batch["x_coronal"].squeeze(0).to(device)
    y_coronal = batch["y_coronal"].squeeze(0).to(device)
    x_sagittal = batch["x_sagittal"].squeeze(0).to(device)
    y_sagittal = batch["y_sagittal"].squeeze(0).to(device)

    def normalize_to_unit_sphere(x):
        """Normalize each 1x3 vector to unit length.
        Args:
            x: tensor of shape (slices, 3, h, w)
        Returns:
            normalized tensor of same shape
        """
        # Reshape to (slices*h*w, 3)
        orig_shape = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
        # Normalize each vector to unit length
        x_norm = F.normalize(x_flat, p=2, dim=1)
        # Reshape back
        x_norm = x_norm.reshape(orig_shape[0], orig_shape[2], orig_shape[3], 3).permute(0, 3, 1, 2)
        return x_norm

    # Normalize all inputs and targets to unit sphere
    x_axial = normalize_to_unit_sphere(x_axial)
    y_axial = normalize_to_unit_sphere(y_axial)
    x_coronal = normalize_to_unit_sphere(x_coronal)
    y_coronal = normalize_to_unit_sphere(y_coronal)
    x_sagittal = normalize_to_unit_sphere(x_sagittal)
    y_sagittal = normalize_to_unit_sphere(y_sagittal)

    # show all incoming shape
    # printlog(f"Incoming shapes:")
    # printlog(f"x_axial: {x_axial.shape}")
    # printlog(f"y_axial: {y_axial.shape}")
    # printlog(f"x_coronal: {x_coronal.shape}")
    # printlog(f"y_coronal: {y_coronal.shape}")
    # printlog(f"x_sagittal: {x_sagittal.shape}")
    # printlog(f"y_sagittal: {y_sagittal.shape}")

    # Ensure spatial dimensions (h, w) are multiple of 16 for each view
    required_multiple = 8
    
    # Pad axial view
    pad_h = (required_multiple - x_axial.shape[2] % required_multiple) % required_multiple
    pad_w = (required_multiple - x_axial.shape[3] % required_multiple) % required_multiple
    if pad_h > 0 or pad_w > 0:
        x_axial = torch.nn.functional.pad(x_axial, (0, pad_w, 0, pad_h), mode='constant', value=0)
        y_axial = torch.nn.functional.pad(y_axial, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    # Pad coronal view
    pad_h = (required_multiple - x_coronal.shape[2] % required_multiple) % required_multiple
    pad_w = (required_multiple - x_coronal.shape[3] % required_multiple) % required_multiple
    if pad_h > 0 or pad_w > 0:
        x_coronal = torch.nn.functional.pad(x_coronal, (0, pad_w, 0, pad_h), mode='constant', value=0)
        y_coronal = torch.nn.functional.pad(y_coronal, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    # Pad sagittal view
    pad_h = (required_multiple - x_sagittal.shape[2] % required_multiple) % required_multiple
    pad_w = (required_multiple - x_sagittal.shape[3] % required_multiple) % required_multiple
    if pad_h > 0 or pad_w > 0:
        x_sagittal = torch.nn.functional.pad(x_sagittal, (0, pad_w, 0, pad_h), mode='constant', value=0)
        y_sagittal = torch.nn.functional.pad(y_sagittal, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # show shapes after padding
    # printlog(f"Shapes after padding:")
    # printlog(f"x_axial: {x_axial.shape}")
    # printlog(f"y_axial: {y_axial.shape}")
    # printlog(f"x_coronal: {x_coronal.shape}")
    # printlog(f"y_coronal: {y_coronal.shape}")
    # printlog(f"x_sagittal: {x_sagittal.shape}")
    # printlog(f"y_sagittal: {y_sagittal.shape}")

    case_loss_axial = 0.0
    case_loss_coronal = 0.0
    case_loss_sagittal = 0.0

    # Process axial view
    indices_list = [i for i in range(0, x_axial.shape[0])]
    random.shuffle(indices_list)
    
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, x_axial.shape[2], x_axial.shape[3])).to(device)
    batch_y = torch.zeros((batch_size, 3, x_axial.shape[2], x_axial.shape[3])).to(device)
    
    for index in indices_list:
        batch_x[batch_size_count] = x_axial[index]
        batch_y[batch_size_count] = y_axial[index]

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list[-1]:
            continue
        else:
            if stage == "train":
                aug_batch_x = batch_x.clone()
                aug_batch_y = batch_y.clone()
                
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    aug_batch_x = torch.rot90(aug_batch_x, k=k, dims=[-2, -1])
                    aug_batch_y = torch.rot90(aug_batch_y, k=k, dims=[-2, -1])
                
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-1])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-1])
                
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-2])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-2])

                case_loss_axial += process_batch(aug_batch_x, aug_batch_y, stage, model, optimizer, device, loss_fn)
            else:
                case_loss_axial += process_batch(batch_x, batch_y, stage, model, optimizer, device, loss_fn)
            batch_size_count = 0

    case_loss_axial = case_loss_axial / (len(indices_list) // batch_size + 1)

    # Process coronal view
    indices_list = [i for i in range(0, x_coronal.shape[0])]
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
                aug_batch_x = batch_x.clone()
                aug_batch_y = batch_y.clone()
                
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    aug_batch_x = torch.rot90(aug_batch_x, k=k, dims=[-2, -1])
                    aug_batch_y = torch.rot90(aug_batch_y, k=k, dims=[-2, -1])
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-1])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-1])
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-2])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-2])

                case_loss_coronal += process_batch(aug_batch_x, aug_batch_y, stage, model, optimizer, device, loss_fn)
            else:
                case_loss_coronal += process_batch(batch_x, batch_y, stage, model, optimizer, device, loss_fn)
            batch_size_count = 0
            
    case_loss_coronal = case_loss_coronal / (len(indices_list) // batch_size + 1)

    # Process sagittal view
    indices_list = [i for i in range(0, x_sagittal.shape[0])]
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
                aug_batch_x = batch_x.clone()
                aug_batch_y = batch_y.clone()
                
                k = random.choice([0, 1, 2, 3])
                if k > 0:
                    aug_batch_x = torch.rot90(aug_batch_x, k=k, dims=[-2, -1])
                    aug_batch_y = torch.rot90(aug_batch_y, k=k, dims=[-2, -1])
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-1])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-1])
                if random.random() < 0.5:
                    aug_batch_x = torch.flip(aug_batch_x, dims=[-2])
                    aug_batch_y = torch.flip(aug_batch_y, dims=[-2])

                case_loss_sagittal += process_batch(aug_batch_x, aug_batch_y, stage, model, optimizer, device, loss_fn)
            else:
                case_loss_sagittal += process_batch(batch_x, batch_y, stage, model, optimizer, device, loss_fn)
            batch_size_count = 0
            
    case_loss_sagittal = case_loss_sagittal / (len(indices_list) // batch_size + 1)

    return case_loss_axial, case_loss_coronal, case_loss_sagittal

def process_batch(batch_x, batch_y, stage, model, optimizer, device, loss_fn=F.mse_loss):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    if stage == "train":
        optimizer.zero_grad()
        loss = model(img=batch_y, cond=batch_x, loss_fn=loss_fn)
        loss.backward()
        optimizer.step()
        return loss.item()
    elif stage == "eval" or stage == "test":
        with torch.no_grad():
            loss = model(img=batch_y, cond=batch_x, loss_fn=loss_fn)
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
    return transforms
