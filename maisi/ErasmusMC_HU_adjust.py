import os
import json
import nibabel as nib
import numpy as np
import argparse
from tqdm import tqdm

# =============================================================================
# Configuration and Global Constants
# =============================================================================

CASE_NAMES = [
    # 'E4058',
    'E4055',          'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139',
]

# Default parameters
DEFAULT_INPUT_DIR = "ErasmusMC"
DEFAULT_OUTPUT_DIR = "ErasmusMC/HUadj"
DEFAULT_SEG_DIR = "ErasmusMC"
DEFAULT_HU_PARAMS = "sCT_CT_stats_no200.npy"
DEFAULT_CONTOUR_TYPE = "bcC"

# HU and intensity boundaries
BODY_CONTOUR_BOUNDARY = -500
MIN_BOUNDARY = -1024
MAX_BOUNDARY = 3000

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

def adjust_hu_values(synCT_data, seg_data, hu_adjustment_params):
    """
    Adjust the HU values of synCT_data using the provided HU adjustment parameters
    and the segmentation data.
    
    Args:
        synCT_data: The synthetic CT data to adjust
        seg_data: The segmentation data with class labels
        hu_adjustment_params: Dictionary with HU adjustment parameters for each class
    
    Returns:
        Adjusted synthetic CT data
    """
    # Create a copy of the data to avoid modifying the original
    adjusted_data = synCT_data.copy()
    
    # Print unique classes in segmentation for debugging
    unique_classes = np.unique(seg_data)
    print(f"  Unique classes in segmentation: {unique_classes}")
    
    for key, stats in hu_adjustment_params.items():
        # Skip if the key is not in the segmentation data
        if int(key) not in unique_classes:
            print(f"  Class {key} not found in segmentation, skipping")
            continue
            
        class_synCT_mean = stats["sCT_mean"]
        class_synCT_std = stats["sCT_std"]
        class_CT_mean = stats["CT_mean"]
        class_CT_std = stats["CT_std"]
        
        # Create mask for this class
        class_mask = seg_data == int(key)
        class_pixels = np.sum(class_mask)
        print(f"  Class {key}: {class_pixels} pixels")
        
        # Apply adjustment formula: (x - mean_synCT) * (std_CT / std_synCT) + mean_CT
        adjusted_data[class_mask] = (
            (synCT_data[class_mask] - class_synCT_mean) * 
            (class_CT_std / class_synCT_std) + 
            class_CT_mean
        )
    
    return adjusted_data

def load_hu_adjustment_params(params_path):
    """
    Load HU adjustment parameters from a file.
    Supports both JSON and NPY formats.
    
    Args:
        params_path: Path to the parameters file
        
    Returns:
        Dictionary with HU adjustment parameters
    """
    if params_path.endswith('.json'):
        with open(params_path, 'r') as f:
            return json.load(f)
    elif params_path.endswith('.npy'):
        return np.load(params_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported file format for HU adjustment parameters: {params_path}")

def process_case(case_name, args):
    """
    Process a single case: load data, adjust HU values, and save results.
    
    Args:
        case_name: Name of the case to process
        args: Command-line arguments
    """
    print(f"Processing case {case_name}...")
    
    # Define file paths
    synCT_path = os.path.join(args.input_dir, f"SynCT_{case_name}_{args.contour_type}.nii.gz")
    seg_path = os.path.join(args.seg_dir, f"{case_name}_combined_MAISI_{args.contour_type}.nii.gz")
    
    # Check if files exist
    if not os.path.exists(synCT_path):
        print(f"  Warning: Synthetic CT file not found: {synCT_path}")
        return
    if not os.path.exists(seg_path):
        print(f"  Warning: Segmentation file not found: {seg_path}")
        return
    
    # Load images
    synCT_img, synCT_data = load_nifti(synCT_path)
    seg_img, seg_data = load_nifti(seg_path)
    
    # Ensure all data has the same shape
    if synCT_data.shape != seg_data.shape:
        print(f"  Warning: Shape mismatch - SynCT: {synCT_data.shape}, Seg: {seg_data.shape}")
        # Match the shapes if needed (this is a simplified approach)
        min_shape = [min(synCT_data.shape[i], seg_data.shape[i]) for i in range(3)]
        synCT_data = synCT_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        seg_data = seg_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Load HU adjustment parameters
    hu_adjustment_params = load_hu_adjustment_params(args.hu_params)
    
    # Adjust HU values
    print(f"  Adjusting HU values...")
    synCT_data = adjust_hu_values(synCT_data, seg_data, hu_adjustment_params)
    
    # Save adjusted synthetic CT
    output_filename = f"SynCT_{case_name}_{args.contour_type}_HUadj.nii.gz"
    output_path = os.path.join(args.output_dir, output_filename)
    save_nifti(synCT_data, synCT_img, output_path)
    print(f"  Saved to {output_path}")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adjust HU values in synthetic CT images')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, 
                        help=f'Directory containing synthetic CT images (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f'Directory to save processed images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--seg_dir', type=str, default=DEFAULT_SEG_DIR, 
                        help=f'Directory containing segmentation images (default: {DEFAULT_SEG_DIR})')
    parser.add_argument('--hu_params', type=str, default=DEFAULT_HU_PARAMS, 
                        help=f'Path to HU adjustment parameters file (default: {DEFAULT_HU_PARAMS})')
    parser.add_argument('--contour_type', type=str, default=DEFAULT_CONTOUR_TYPE, 
                        help=f'Body contour type (default: {DEFAULT_CONTOUR_TYPE})')
    parser.add_argument('--case_id', type=str, default=None, 
                        help='Process specific case ID (default: process all cases)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("Running with the following configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Segmentation directory: {args.seg_dir}")
    print(f"  HU parameters file: {args.hu_params}")
    print(f"  Contour type: {args.contour_type}")
    
    # Determine which cases to process
    if args.case_id:
        cases_to_process = [args.case_id]
        print(f"Processing single case: {args.case_id}")
    else:
        # Process all cases in the predefined list
        cases_to_process = CASE_NAMES
        print(f"Processing all {len(cases_to_process)} cases from the predefined list")
    
    # Process each case
    for case_name in tqdm(cases_to_process):
        process_case(case_name, args)
    
    print("All processing completed.")

if __name__ == "__main__":
    main()
