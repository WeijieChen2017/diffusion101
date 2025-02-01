import os
import json
import nibabel as nib
import numpy as np
from scipy.ndimage import sobel, binary_fill_holes
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# =============================================================================
# Configuration and Global Constants
# =============================================================================

CASE_NAMES = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139',
]
REGIONS = ["body", "soft", "bone"]

# Directories
ROOT_DIR = "James_36/synCT"
CT_DIR = "NAC_CTAC_Spacing15"
MASK_DIR = "James_36/CT_mask"
SYNCT_DIR = os.path.join(ROOT_DIR, "inference_20250128_noon")
SYNCT_SEG_DIR = ROOT_DIR
SAVE_DIR = ROOT_DIR


# HU and intensity boundaries
BODY_CONTOUR_BOUNDARY = -500
MIN_BOUNDARY = -1024
SOFT_BOUNDARY = -500
BONE_BOUNDARY = 300
MAX_BOUNDARY = 3000

# Flags
CT_MASK_OVERWRITE = False
PRED_MASK_OVERWRITE = True
HU_ADJUSTMENT_ENABLED = True

# Load HU adjustment parameters
HU_ADJUSTMENT_PATH = "sCT_CT_stats.npy"
HU_VALUE_ADJUSTMENT = np.load(HU_ADJUSTMENT_PATH, allow_pickle=True).item()

# Initialize a dictionary for metrics
metrics_dict = {
    "mae_by_case": {},
    "mae_by_region": {},
    "ssim_by_case": {},
    "ssim_by_region": {},
    "psnr_by_case": {},
    "psnr_by_region": {},
    "dsc_by_case": {},
    "dsc_by_region": {},
    "acutance_by_case": {},
    "acutance_by_region": {},
}

# =============================================================================
# Helper Functions
# =============================================================================

def load_nifti(file_path):
    """Load a NIfTI file and return the image object and its data array."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data

def save_nifti(data, reference_img, output_path):
    """Save data as a NIfTI file using a reference image for affine and header."""
    new_img = nib.Nifti1Image(data.astype(np.float32), reference_img.affine, reference_img.header)
    nib.save(new_img, output_path)

def match_slices(data, target_slices, pad_value):
    """
    Pad or crop the third dimension (slices) of data so that it matches target_slices.
    """
    current_slices = data.shape[2]
    if current_slices < target_slices:
        pad_width = ((0, 0), (0, 0), (0, target_slices - current_slices))
        data = np.pad(data, pad_width, mode="constant", constant_values=pad_value)
    elif current_slices > target_slices:
        data = data[:, :, :target_slices]
    return data

def generate_body_mask(ct_data):
    """Generate the body mask by thresholding and filling holes slice‐by‐slice."""
    mask = ct_data >= BODY_CONTOUR_BOUNDARY
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])
    return mask.astype(np.uint8)

def generate_soft_mask(ct_data):
    """Generate the soft tissue mask based on HU thresholds."""
    mask = (ct_data >= SOFT_BOUNDARY) & (ct_data <= BONE_BOUNDARY)
    return mask.astype(np.uint8)

def generate_bone_mask(ct_data):
    """Generate the bone mask based on HU thresholds."""
    mask = (ct_data >= BONE_BOUNDARY) & (ct_data <= MAX_BOUNDARY)
    return mask.astype(np.uint8)

def get_or_create_ct_masks(case_name, ct_data, ct_img):
    """
    Get or create the CT masks (body, soft, bone) for a given case.
    If the masks already exist and CT_MASK_OVERWRITE is False, they will be loaded.
    """
    body_mask_path = os.path.join(MASK_DIR, f"mask_body_contour_{case_name}.nii.gz")
    soft_mask_path = os.path.join(MASK_DIR, f"mask_body_soft_{case_name}.nii.gz")
    bone_mask_path = os.path.join(MASK_DIR, f"mask_body_bone_{case_name}.nii.gz")
    
    if os.path.exists(soft_mask_path) and not CT_MASK_OVERWRITE:
        body_mask = nib.load(body_mask_path).get_fdata()
        soft_mask = nib.load(soft_mask_path).get_fdata()
        bone_mask = nib.load(bone_mask_path).get_fdata()
    else:
        body_mask = generate_body_mask(ct_data)
        soft_mask = generate_soft_mask(ct_data)
        bone_mask = generate_bone_mask(ct_data)
        save_nifti(body_mask, ct_img, body_mask_path)
        save_nifti(soft_mask, ct_img, soft_mask_path)
        save_nifti(bone_mask, ct_img, bone_mask_path)
    return body_mask.astype(bool), soft_mask.astype(bool), bone_mask.astype(bool)

def adjust_hu_values(synCT_data, seg_data):
    """
    Adjust the HU values of synCT_data using the HU_VALUE_ADJUSTMENT parameters
    and the provided segmentation data.
    """
    for key, stats in HU_VALUE_ADJUSTMENT.items():
        class_synCT_mean = stats["sCT_mean"]
        class_synCT_std = stats["sCT_std"]
        class_CT_mean = stats["CT_mean"]
        class_CT_std = stats["CT_std"]
        class_mask = seg_data == key
        synCT_data[class_mask] = (synCT_data[class_mask] - class_synCT_mean) * class_CT_std / class_synCT_std + class_CT_mean
    return synCT_data

def get_or_create_pred_masks(case_name, synCT_data, ct_img, ct_data):
    """
    Get or create predicted masks from synCT_data.
    Returns binary masks for body, soft, and bone regions.
    """
    pred_body_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_body_adjusted.nii.gz")
    pred_soft_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_sof_adjusted.nii.gz")
    pred_bone_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_bone_adjusted.nii.gz")
    
    # Process body contour mask
    if os.path.exists(pred_body_path) and not PRED_MASK_OVERWRITE:
        pred_body = nib.load(pred_body_path).get_fdata()
    else:
        pred_body = synCT_data >= BODY_CONTOUR_BOUNDARY
        for i in range(pred_body.shape[2]):
            pred_body[:, :, i] = binary_fill_holes(pred_body[:, :, i])
        save_nifti(pred_body.astype(np.uint8), ct_img, pred_body_path)
    pred_body = match_slices(pred_body, ct_data.shape[2], pad_value=0).astype(bool)
    
    # Process soft and bone masks
    if os.path.exists(pred_soft_path) and not PRED_MASK_OVERWRITE:
        pred_soft = nib.load(pred_soft_path).get_fdata()
        pred_bone = nib.load(pred_bone_path).get_fdata()
    else:
        pred_soft = (synCT_data >= SOFT_BOUNDARY) & (synCT_data <= BONE_BOUNDARY)
        pred_bone = (synCT_data >= BONE_BOUNDARY) & (synCT_data <= MAX_BOUNDARY)
        save_nifti(pred_soft.astype(np.uint8), ct_img, pred_soft_path)
        save_nifti(pred_bone.astype(np.uint8), ct_img, pred_bone_path)
    return pred_body, pred_soft, pred_bone

def normalize_image(data, min_val, max_val):
    """
    Clip the data to [min_val, max_val] and shift it so that the minimum becomes zero.
    """
    data = np.clip(data, min_val, max_val)
    return data - min_val

def compute_region_metrics(ct_data, synCT_data, mask, pred_mask, acutance_data, data_range):
    """
    Compute and return MAE, SSIM, PSNR, DSC, and acutance for a given region mask.
    """
    mae = np.mean(np.abs(ct_data[mask] - synCT_data[mask]))
    ssim_val = ssim(ct_data[mask], synCT_data[mask], data_range=data_range)
    psnr_val = psnr(ct_data[mask], synCT_data[mask], data_range=data_range)
    intersection = np.sum(mask & pred_mask)
    dsc = 2 * intersection / (np.sum(mask) + np.sum(pred_mask) + 1e-8)
    acutance = np.mean(acutance_data[mask])
    return mae, ssim_val, psnr_val, dsc, acutance

def process_case(case_name):
    """Process a single case: load data, adjust HU (if enabled), generate masks, and compute metrics."""
    print(f"Processing case {case_name}...")
    # Define file paths
    ct_path = os.path.join(CT_DIR, f"CTAC_{case_name}_cropped.nii.gz")
    synCT_path = os.path.join(SYNCT_DIR, f"CTAC_{case_name}_TS_MAISI.nii.gz")
    synCT_seg_path = os.path.join(ROOT_DIR, f"SynCT_{case_name}_TS_label.nii.gz")
    
    # Load images
    ct_img, ct_data = load_nifti(ct_path)
    synCT_img, synCT_data = load_nifti(synCT_path)
    
    # If HU adjustment is enabled, load the segmentation and adjust synCT_data
    if HU_ADJUSTMENT_ENABLED:
        _, synCT_seg_data = load_nifti(synCT_seg_path)
        synCT_seg_data = match_slices(synCT_seg_data, ct_data.shape[2], pad_value=0)
    
    # Match the number of slices of synCT_data with ct_data
    synCT_data = match_slices(synCT_data, ct_data.shape[2], pad_value=MIN_BOUNDARY)
    
    # Create or load the CT masks
    body_mask, soft_mask, bone_mask = get_or_create_ct_masks(case_name, ct_data, ct_img)
    
    # HU adjustment (if enabled)
    if HU_ADJUSTMENT_ENABLED:
        synCT_data = adjust_hu_values(synCT_data, synCT_seg_data)
        adjusted_path = synCT_path.replace(".nii.gz", "_adjusted.nii.gz")
        save_nifti(synCT_data, synCT_img, adjusted_path)
    
    # Create or load predicted masks from synCT_data
    pred_body, pred_soft, pred_bone = get_or_create_pred_masks(case_name, synCT_data, ct_img, ct_data)
    
    # Compute the image gradient (acutance) using the Sobel filter
    acutance_whole = np.abs(sobel(synCT_data))
    
    # Normalize the images (both CT and synCT) to the same range
    ct_data_norm = normalize_image(ct_data, MIN_BOUNDARY, MAX_BOUNDARY)
    synCT_data_norm = normalize_image(synCT_data, MIN_BOUNDARY, MAX_BOUNDARY)
    data_range = MAX_BOUNDARY - MIN_BOUNDARY
    
    # Prepare lists of ground truth and predicted masks
    gt_masks = [body_mask, soft_mask, bone_mask]
    pred_masks = [pred_body, pred_soft, pred_bone]
    
    # Initialize the case entry in the metrics dictionary
    if case_name not in metrics_dict["mae_by_case"]:
        metrics_dict["mae_by_case"][case_name] = {}
        metrics_dict["ssim_by_case"][case_name] = {}
        metrics_dict["psnr_by_case"][case_name] = {}
        metrics_dict["dsc_by_case"][case_name] = {}
        metrics_dict["acutance_by_case"][case_name] = {}
    
    # Compute metrics for each region
    for region, gt_mask, pred_mask in zip(REGIONS, gt_masks, pred_masks):
        mae, ssim_val, psnr_val, dsc, acutance = compute_region_metrics(
            ct_data_norm, synCT_data_norm, gt_mask, pred_mask, acutance_whole, data_range
        )
        print(f"  {region.capitalize()}: MAE={mae:.4f}, SSIM={ssim_val:.4f}, PSNR={psnr_val:.4f}, DSC={dsc:.4f}, Acutance={acutance:.4f}")
        
        metrics_dict["mae_by_case"][case_name][region] = mae
        metrics_dict["ssim_by_case"][case_name][region] = ssim_val
        metrics_dict["psnr_by_case"][case_name][region] = psnr_val
        metrics_dict["dsc_by_case"][case_name][region] = dsc
        metrics_dict["acutance_by_case"][case_name][region] = acutance
        
        # Append the metrics to region-level lists for averaging later
        if region not in metrics_dict["mae_by_region"]:
            metrics_dict["mae_by_region"][region] = []
            metrics_dict["ssim_by_region"][region] = []
            metrics_dict["psnr_by_region"][region] = []
            metrics_dict["dsc_by_region"][region] = []
            metrics_dict["acutance_by_region"][region] = []
        metrics_dict["mae_by_region"][region].append(mae)
        metrics_dict["ssim_by_region"][region].append(ssim_val)
        metrics_dict["psnr_by_region"][region].append(psnr_val)
        metrics_dict["dsc_by_region"][region].append(dsc)
        metrics_dict["acutance_by_region"][region].append(acutance)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    for case_name in CASE_NAMES:
        process_case(case_name)
    
    # Compute average metrics for each region across all cases
    for region in REGIONS:
        metrics_dict["mae_by_region"][region] = np.mean(metrics_dict["mae_by_region"][region])
        metrics_dict["ssim_by_region"][region] = np.mean(metrics_dict["ssim_by_region"][region])
        metrics_dict["psnr_by_region"][region] = np.mean(metrics_dict["psnr_by_region"][region])
        metrics_dict["dsc_by_region"][region] = np.mean(metrics_dict["dsc_by_region"][region])
        metrics_dict["acutance_by_region"][region] = np.mean(metrics_dict["acutance_by_region"][region])
    
    print("\nAverage metrics by region:")
    print(f"MAE: {metrics_dict['mae_by_region']}")
    print(f"SSIM: {metrics_dict['ssim_by_region']}")
    print(f"PSNR: {metrics_dict['psnr_by_region']}")
    print(f"DSC: {metrics_dict['dsc_by_region']}")
    print(f"Acutance: {metrics_dict['acutance_by_region']}")
    
    print("\nMetrics by case:")
    print(f"MAE: {metrics_dict['mae_by_case']}")
    print(f"SSIM: {metrics_dict['ssim_by_case']}")
    print(f"PSNR: {metrics_dict['psnr_by_case']}")
    print(f"DSC: {metrics_dict['dsc_by_case']}")
    print(f"Acutance: {metrics_dict['acutance_by_case']}")
    
    # Save the metrics to a JSON file
    metrics_json_path = os.path.join(ROOT_DIR, "LDM36v2_metrics_adjusted.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nSaved metrics to {metrics_json_path}")
    print("Done!")

if __name__ == "__main__":
    main()
