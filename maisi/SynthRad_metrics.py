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
    "1PA001", "1PA024", "1PA047", "1PA065", "1PA091", "1PA110", "1PA133", "1PA150", "1PA168", "1PA185", "1PC018", "1PC039", "1PC058", "1PC080",
    "1PA004", "1PA025", "1PA048", "1PA070", "1PA093", "1PA111", "1PA134", "1PA151", "1PA169", "1PA187", "1PC019", "1PC040", "1PC059", "1PC082",
    "1PA005", "1PA026", "1PA049", "1PA073", "1PA094", "1PA112", "1PA136", "1PA152", "1PA170", "1PA188", "1PC022", "1PC041", "1PC061", "1PC084",
    "1PA009", "1PA028", "1PA052", "1PA074", "1PA095", "1PA113", "1PA137", "1PA154", "1PA171", "1PC000", "1PC023", "1PC042", "1PC063", "1PC085",
    "1PA010", "1PA029", "1PA053", "1PA076", "1PA097", "1PA114", "1PA138", "1PA155", "1PA173", "1PC001", "1PC024", "1PC044", "1PC065", "1PC088",
    "1PA011", "1PA030", "1PA054", "1PA079", "1PA098", "1PA115", "1PA140", "1PA156", "1PA174", "1PC004", "1PC027", "1PC045", "1PC066", "1PC092",
    "1PA012", "1PA031", "1PA056", "1PA080", "1PA100", "1PA116", "1PA141", "1PA157", "1PA176", "1PC006", "1PC029", "1PC046", "1PC069", "1PC093",
    "1PA014", "1PA035", "1PA058", "1PA081", "1PA101", "1PA117", "1PA142", "1PA159", "1PA177", "1PC007", "1PC032", "1PC048", "1PC070", "1PC095",
    "1PA018", "1PA038", "1PA059", "1PA083", "1PA105", "1PA118", "1PA144", "1PA161", "1PA178", "1PC010", "1PC033", "1PC049", "1PC071", "1PC096",
    "1PA019", "1PA040", "1PA060", "1PA084", "1PA106", "1PA119", "1PA145", "1PA163", "1PA180", "1PC011", "1PC035", "1PC052", "1PC073", "1PC097",
    "1PA020", "1PA041", "1PA062", "1PA086", "1PA107", "1PA121", "1PA146", "1PA164", "1PA181", "1PC013", "1PC036", "1PC054", "1PC077", "1PC098",
    "1PA021", "1PA044", "1PA063", "1PA088", "1PA108", "1PA126", "1PA147", "1PA165", "1PA182", "1PC015", "1PC037", "1PC055", "1PC078",
    "1PA022", "1PA045", "1PA064", "1PA090", "1PA109", "1PA127", "1PA148", "1PA167", "1PA183", "1PC017", "1PC038", "1PC057", "1PC079",
]
REGIONS = ["body", "soft", "bone"]

# Directories
ROOT_DIR = "SynthRad_nifti"
CT_DIR = f"{ROOT_DIR}/ct"
MASK_DIR = f"{ROOT_DIR}/mask/"
SYNCT_DIR = f"{ROOT_DIR}/synCT_label_painting"
SYNCT_SEG_DIR = f"{ROOT_DIR}/label_painting"
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
HU_ADJUSTMENT_PATH = "SynthRad_CT_stats.npy"
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
    if HU_ADJUSTMENT_ENABLED:
        pred_body_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_body_adjusted.nii.gz")
        pred_soft_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_sof_adjusted.nii.gz")
        pred_bone_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_bone_adjusted.nii.gz")
    else:
        pred_body_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_body.nii.gz")
        pred_soft_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_soft.nii.gz")
        pred_bone_path = os.path.join(SYNCT_SEG_DIR, f"SynCT_{case_name}_TS_mask_bone.nii.gz")
    
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
    ct_path = os.path.join(CT_DIR, f"{case_name}_ct.nii.gz")
    synCT_path = os.path.join(SYNCT_DIR, f"{case_name}_bg.nii.gz")
    synCT_seg_path = os.path.join(SYNCT_SEG_DIR, f"{case_name}_label_painting.nii.gz")
    
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
        adjusted_path = synCT_path.replace(".nii.gz", "_adjusted_vanilaoverlap.nii.gz")
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
    # metrics_json_path = os.path.join(ROOT_DIR, "LDM36v2_metrics_adjusted.json")
    metrics_json_path = os.path.join(ROOT_DIR, "SynthRad_adjusted.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nSaved metrics to {metrics_json_path}")
    print("Done!")

if __name__ == "__main__":
    main()
