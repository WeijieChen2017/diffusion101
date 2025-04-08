import os
import argparse
import json
import time
import random
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import sobel

from LDM_utils import VQModel

# Set the environment variable to use the GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device is: ", device)

# Constants for normalization
HU_MIN = -1024
HU_MAX = 1976
RANGE_HU = HU_MAX - HU_MIN

# Constants for mask generation
BODY_CONTOUR_BOUNDARY = -500
MIN_BOUNDARY = -1024
SOFT_BOUNDARY = -500
BONE_BOUNDARY = 150
MAX_BOUNDARY = 1976

def normalize(tensor, hu_min=HU_MIN, hu_max=HU_MAX):
    """Normalize the input tensor to [0, 1] range"""
    # Clip to the HU range
    tensor_clipped = np.clip(tensor, hu_min, hu_max)
    # Scale to [0, 1]
    return (tensor_clipped - hu_min) / (hu_max - hu_min)

def denormalize(tensor, hu_min=HU_MIN, hu_max=HU_MAX):
    """Denormalize the input tensor from [0, 1] range back to HU range"""
    return tensor * (hu_max - hu_min) + hu_min

def normalize_residual(residual, hu_min=HU_MIN, hu_max=HU_MAX):
    """Normalize the residual to [-1, 1] range"""
    range_hu = hu_max - hu_min
    return 2.0 * residual / range_hu

def denormalize_residual(normalized_residual, hu_min=HU_MIN, hu_max=HU_MAX):
    """Denormalize the residual from [-1, 1] range back to HU range"""
    range_hu = hu_max - hu_min
    return normalized_residual * range_hu / 2.0

def setup_logger(log_dir):
    """Set up a logger that writes to a text file"""
    import datetime
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_log_{timestamp}.txt")
    
    def log_message(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_entry)
        
        # Write to log file
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
    
    return log_message

def generate_masks(ct_data, output_dir, case_name, ct_file, overwrite=False):
    """Generate body, soft tissue, and bone masks from CT data"""
    body_mask_path = os.path.join(output_dir, f"{case_name}_body.nii.gz")
    soft_mask_path = os.path.join(output_dir, f"{case_name}_soft.nii.gz")
    bone_mask_path = os.path.join(output_dir, f"{case_name}_bone.nii.gz")
    
    # If masks already exist and not overwriting, load them
    if os.path.exists(soft_mask_path) and not overwrite:
        body_mask = nib.load(body_mask_path).get_fdata()
        soft_mask = nib.load(soft_mask_path).get_fdata()
        bone_mask = nib.load(bone_mask_path).get_fdata()
    else:
        # Generate body contour mask
        body_mask = ct_data >= BODY_CONTOUR_BOUNDARY
        for i in range(body_mask.shape[2]):
            body_mask[:, :, i] = binary_fill_holes(body_mask[:, :, i])
        
        # Generate soft tissue mask
        soft_mask = (ct_data >= SOFT_BOUNDARY) & (ct_data <= BONE_BOUNDARY)
        
        # Generate bone mask
        bone_mask = (ct_data >= BONE_BOUNDARY) & (ct_data <= MAX_BOUNDARY)
        
        # Save masks
        body_mask_nii = nib.Nifti1Image(body_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(body_mask_nii, body_mask_path)
        
        soft_mask_nii = nib.Nifti1Image(soft_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(soft_mask_nii, soft_mask_path)
        
        bone_mask_nii = nib.Nifti1Image(bone_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(bone_mask_nii, bone_mask_path)
    
    return body_mask, soft_mask, bone_mask

def process_volume(model, volume, logger=None):
    """
    Process a volume from all three views (axial, coronal, sagittal)
    and return the merged prediction and individual view predictions.
    """
    # Get original shape for later cropping
    original_shape = volume.shape
    
    # Pad volume to make dimensions divisible by 4
    pad_z, pad_y, pad_x = 0, 0, 0
    if volume.shape[0] % 4 != 0:
        pad_z = 4 - volume.shape[0] % 4
    if volume.shape[1] % 4 != 0:
        pad_y = 4 - volume.shape[1] % 4
    if volume.shape[2] % 4 != 0:
        pad_x = 4 - volume.shape[2] % 4
    
    # Create padded volume
    padded_volume = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    
    # Prepare empty arrays for predictions
    axial_pred = np.zeros_like(padded_volume)
    coronal_pred = np.zeros_like(padded_volume)
    sagittal_pred = np.zeros_like(padded_volume)
    
    # Process axial slices
    if logger:
        logger("Processing axial slices...")
    for z in range(1, padded_volume.shape[0] - 1):
        # Input context: 3 adjacent slices
        x = padded_volume[z-1:z+2, :, :]  # 3 x H x W
        x = np.transpose(x, (1, 2, 0))    # H x W x 3
        
        # Input center slice for residual addition
        x_center = padded_volume[z, :, :]  # H x W
        
        # Normalize input
        x_norm = normalize(x)
        x_center_norm = normalize(x_center)
        
        # Convert to tensor and add batch dimension
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        x_center_tensor = torch.tensor(x_center_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            residual = model(x_tensor)
            residual = torch.clamp(residual, -1.0, 1.0)
            prediction_norm = x_center_tensor + residual
            prediction_norm = torch.clamp(prediction_norm, 0.0, 1.0)
            
            # Convert back to numpy
            prediction = prediction_norm.cpu().numpy().squeeze()
            
            # Store in the prediction volume
            axial_pred[z, :, :] = prediction
    
    # Process coronal slices
    if logger:
        logger("Processing coronal slices...")
    for y in range(1, padded_volume.shape[1] - 1):
        # Input context: 3 adjacent slices
        x = padded_volume[:, y-1:y+2, :]  # Z x 3 x W
        x = np.transpose(x, (0, 2, 1))    # Z x W x 3
        
        # Input center slice for residual addition
        x_center = padded_volume[:, y, :]  # Z x W
        
        # Normalize input
        x_norm = normalize(x)
        x_center_norm = normalize(x_center)
        
        # Convert to tensor and add batch dimension
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        x_center_tensor = torch.tensor(x_center_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            residual = model(x_tensor)
            residual = torch.clamp(residual, -1.0, 1.0)
            prediction_norm = x_center_tensor + residual
            prediction_norm = torch.clamp(prediction_norm, 0.0, 1.0)
            
            # Convert back to numpy
            prediction = prediction_norm.cpu().numpy().squeeze()
            
            # Store in the prediction volume
            coronal_pred[:, y, :] = prediction
    
    # Process sagittal slices
    if logger:
        logger("Processing sagittal slices...")
    for x in range(1, padded_volume.shape[2] - 1):
        # Input context: 3 adjacent slices
        slices = padded_volume[:, :, x-1:x+2]  # Z x H x 3
        
        # Input center slice for residual addition
        x_center = padded_volume[:, :, x]  # Z x H
        
        # Normalize input
        x_norm = normalize(slices)
        x_center_norm = normalize(x_center)
        
        # Convert to tensor and add batch dimension
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        x_center_tensor = torch.tensor(x_center_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            residual = model(x_tensor)
            residual = torch.clamp(residual, -1.0, 1.0)
            prediction_norm = x_center_tensor + residual
            prediction_norm = torch.clamp(prediction_norm, 0.0, 1.0)
            
            # Convert back to numpy
            prediction = prediction_norm.cpu().numpy().squeeze()
            
            # Store in the prediction volume
            sagittal_pred[:, :, x] = prediction
    
    # Combine predictions using median
    if logger:
        logger("Combining predictions using median...")
    
    # Convert normalized predictions back to HU range
    axial_pred = denormalize(axial_pred)
    coronal_pred = denormalize(coronal_pred)
    sagittal_pred = denormalize(sagittal_pred)
    
    # Stack predictions and compute median
    predictions = np.stack([axial_pred, coronal_pred, sagittal_pred], axis=0)
    merged_pred = np.median(predictions, axis=0)
    
    # Crop back to original size
    original_pred = merged_pred[:original_shape[0], :original_shape[1], :original_shape[2]]
    original_axial = axial_pred[:original_shape[0], :original_shape[1], :original_shape[2]]
    original_coronal = coronal_pred[:original_shape[0], :original_shape[1], :original_shape[2]]
    original_sagittal = sagittal_pred[:original_shape[0], :original_shape[1], :original_shape[2]]
    
    return original_pred, original_axial, original_coronal, original_sagittal

def calculate_metrics(ct_data, pred_data, masks, region_names, logger=None):
    """Calculate only MAE for different regions using masks"""
    metrics = {}
    
    for i, region in enumerate(region_names):
        gt_mask = masks[i]
        
        # Calculate MAE only
        mae = np.mean(np.abs(ct_data[gt_mask] - pred_data[gt_mask]))
        
        metrics[region] = {
            'mae': mae
        }
        
        if logger:
            logger(f"Region {region}: MAE={mae:.4f} HU")
    
    return metrics

def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description='Run inference on test cases')
    argparser.add_argument('--cross_validation', type=int, default=1, help='Index of the cross validation fold')
    argparser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    argparser.add_argument('--test_cases', type=str, default='LDM_adapter/LDM_folds_with_test.json', help='JSON file with test cases')
    argparser.add_argument('--data_dir', type=str, default='LDM_adapter', help='Root directory containing the data')
    argparser.add_argument('--output_dir', type=str, default='LDM_adapter/results/predictions', help='Directory to save predictions')
    argparser.add_argument('--mask_dir', type=str, default='LDM_adapter/results/masks', help='Directory to save masks')
    argparser.add_argument('--metrics_dir', type=str, default='LDM_adapter/results/metrics', help='Directory to save metrics')
    argparser.add_argument('--log_dir', type=str, default='LDM_adapter/results/inference_logs', help='Directory to save logs')
    argparser.add_argument('--overwrite_masks', action='store_false', help='Overwrite existing masks')
    args = argparser.parse_args()
    
    # Create fold-specific output directories
    fold_suffix = f"fold_{args.cross_validation}"
    args.output_dir = os.path.join(args.output_dir, fold_suffix)
    args.mask_dir = os.path.join(args.mask_dir, fold_suffix)
    args.metrics_dir = os.path.join(args.metrics_dir, fold_suffix)
    args.log_dir = os.path.join(args.log_dir, fold_suffix)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Set checkpoint path based on cross-validation fold if not manually specified
    if args.checkpoint is None:
        args.checkpoint = f'results/fold_{args.cross_validation}/best_model_fold_{args.cross_validation}.pth'
        logger(f"Checkpoint path automatically set to: {args.checkpoint}")
    
    # Set random seed for reproducibility
    random_seed = 729
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger(f"Starting inference with fold {args.cross_validation}")
    logger(f"Random seed: {random_seed}")
    logger(f"Using checkpoint: {args.checkpoint}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Load test cases from JSON
    with open(args.test_cases, 'r') as f:
        folds_data = json.load(f)
    
    # Extract test cases for the specified fold
    test_cases = folds_data[f"fold_{args.cross_validation}"]["test"]
    logger(f"Found {len(test_cases)} test cases for fold {args.cross_validation}")
    
    # Initialize model
    model_params = {
        "VQ_NAME": "ldm-residual-model",
        "n_embed": 8192,
        "embed_dim": 3,
        "img_size": 256,
        "ddconfig": {
            "attn_type": "none",
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,  # Input has 3 channels (context slices)
            "out_ch": 1,       # Output has 1 channel (residual)
            "ch": 128,         # Base channel count
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
    }
    
    logger("Initializing model...")
    model = VQModel(
        ddconfig=model_params["ddconfig"],
        n_embed=model_params["n_embed"],
        embed_dim=model_params["embed_dim"],
    )
    
    # Load model checkpoint
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        logger(f"Model weights loaded from {args.checkpoint}")
    except Exception as e:
        logger(f"Error loading model weights: {str(e)}")
        return
    
    model.to(device)
    model.eval()
    
    # Prepare for storing metrics
    region_names = ["body", "soft", "bone"]
    all_metrics = {region: {
        'mae': []
    } for region in region_names}
    
    case_metrics = {}
    
    # Process each test case
    logger(f"Processing {len(test_cases)} test volumes...")
    
    for idx, case_id in enumerate(test_cases):
        logger(f"Processing case {idx+1}/{len(test_cases)}: {case_id}")
        
        # Construct paths for input and reference volumes using the format from LDM_utils.py
        ct_path = f"{args.data_dir}/data/CT/CTAC_{case_id}_cropped.nii.gz"
        sct_path = f"{args.data_dir}/data/sCT/CTAC_{case_id}_TS_MAISI.nii.gz"
        
        # Load nifti files directly
        try:
            sct_nifti = nib.load(sct_path)
            ct_nifti = nib.load(ct_path)
            
            # Get input volume, affine, and header
            input_volume = sct_nifti.get_fdata()
            reference_volume = ct_nifti.get_fdata()
            affine = sct_nifti.affine
            header = sct_nifti.header
            
            logger(f"Loaded input volume shape: {input_volume.shape}, reference volume shape: {reference_volume.shape}")
        except Exception as e:
            logger(f"Error loading case {case_id}: {str(e)}")
            continue
        
        # Generate masks for the reference CT volume
        body_mask, soft_mask, bone_mask = generate_masks(
            reference_volume, args.mask_dir, case_id, ct_nifti, 
            overwrite=args.overwrite_masks
        )
        masks = [body_mask, soft_mask, bone_mask]
        
        # Process the volume
        start_time = time.time()
        predicted_volume, axial_volume, coronal_volume, sagittal_volume = process_volume(model, input_volume, logger)
        inference_time = time.time() - start_time
        
        # Ensure the predicted volume matches the original input shape
        if predicted_volume.shape != input_volume.shape:
            logger(f"Warning: Predicted volume shape {predicted_volume.shape} does not match input shape {input_volume.shape}")
            predicted_volume = predicted_volume[:input_volume.shape[0], :input_volume.shape[1], :input_volume.shape[2]]
            axial_volume = axial_volume[:input_volume.shape[0], :input_volume.shape[1], :input_volume.shape[2]]
            coronal_volume = coronal_volume[:input_volume.shape[0], :input_volume.shape[1], :input_volume.shape[2]]
            sagittal_volume = sagittal_volume[:input_volume.shape[0], :input_volume.shape[1], :input_volume.shape[2]]
        
        # Calculate metrics for different regions (only MAE)
        metrics = calculate_metrics(reference_volume, predicted_volume, masks, region_names, logger)
        case_metrics[case_id] = metrics
        
        # Add metrics to overall statistics
        for region in region_names:
            all_metrics[region]['mae'].append(metrics[region]['mae'])
        
        logger(f"Case {case_id} - Inference time: {inference_time:.2f}s")
        
        # Save the merged prediction as a NIfTI file
        output_path = os.path.join(args.output_dir, f"{case_id}_merged.nii.gz")
        nib_img = nib.Nifti1Image(predicted_volume, affine, header)
        nib.save(nib_img, output_path)
        logger(f"Merged prediction saved to {output_path}")
        
        # Save the axial prediction as a NIfTI file
        axial_output_path = os.path.join(args.output_dir, f"{case_id}_axial.nii.gz")
        axial_nib_img = nib.Nifti1Image(axial_volume, affine, header)
        nib.save(axial_nib_img, axial_output_path)
        logger(f"Axial prediction saved to {axial_output_path}")
        
        # Save the coronal prediction as a NIfTI file
        coronal_output_path = os.path.join(args.output_dir, f"{case_id}_coronal.nii.gz")
        coronal_nib_img = nib.Nifti1Image(coronal_volume, affine, header)
        nib.save(coronal_nib_img, coronal_output_path)
        logger(f"Coronal prediction saved to {coronal_output_path}")
        
        # Save the sagittal prediction as a NIfTI file
        sagittal_output_path = os.path.join(args.output_dir, f"{case_id}_sagittal.nii.gz")
        sagittal_nib_img = nib.Nifti1Image(sagittal_volume, affine, header)
        nib.save(sagittal_nib_img, sagittal_output_path)
        logger(f"Sagittal prediction saved to {sagittal_output_path}")
    
    # Calculate average metrics across all cases
    avg_metrics = {region: {
        'mae': np.mean(all_metrics[region]['mae'])
    } for region in region_names}
    
    # Print average metrics
    logger("Average metrics across all cases:")
    for region in region_names:
        logger(f"Region {region}: MAE={avg_metrics[region]['mae']:.4f} HU")
    
    # Save metrics to JSON file
    metrics_output = {
        "metrics_by_case": case_metrics,
        "average_metrics": avg_metrics
    }
    
    metrics_path = os.path.join(args.metrics_dir, f"metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger(f"Metrics saved to {metrics_path}")
    logger("Inference completed successfully!")

if __name__ == "__main__":
    main() 