import nibabel as nib
import numpy as np
import argparse
import os

NAC_TS_label_path = "combined_predictions/E4058_combined.nii.gz" # no 200
NAC_TS_body_contour_PET_path = "NAC_body_contour_thresholds/NAC_E4058_final_contour.nii.gz"
NAC_TS_body_contour_CT_path = "James_36/CT_mask/mask_body_contour_E4058.nii.gz"
CT_TS_label_path = "NAC_CTAC_Spacing15/CTAC_E4058_TS.nii.gz"

# Default output directory
DEFAULT_OUTPUT_DIR = "ErasmusMC"

# Overall Dice (all foreground vs background): 0.8680

# Mapping dictionary from TS to MAISI
T2M_mapping = {
    1: 3,
    2: 5,
    3: 14,
    4: 10,
    5: 1,
    6: 12,
    7: 4,
    8: 8,
    9: 9,
    10: 28,
    11: 29,
    12: 30,
    13: 31,
    14: 32,
    15: 11,
    16: 57,
    17: 126,
    18: 19,
    19: 13,
    20: 62,
    21: 15,
    22: 118,
    23: 116,
    24: 117,
    25: 97,
    26: 127,
    27: 33,
    28: 34,
    29: 35,
    30: 36,
    31: 37,
    32: 38,
    33: 39,
    34: 40,
    35: 41,
    36: 42,
    37: 43,
    38: 44,
    39: 45,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 115,
    52: 6,
    53: 119,
    54: 109,
    55: 123,
    56: 124,
    57: 112,
    58: 113,
    59: 110,
    60: 111,
    61: 108,
    62: 125,
    63: 7,
    64: 17,
    65: 58,
    66: 59,
    67: 60,
    68: 61,
    69: 87,
    70: 88,
    71: 89,
    72: 90,
    73: 91,
    74: 92,
    75: 93,
    76: 94,
    77: 95,
    78: 96,
    79: 121,
    80: 98,
    81: 99,
    82: 100,
    83: 101,
    84: 102,
    85: 103,
    86: 104,
    87: 105,
    88: 106,
    89: 107,
    90: 22,
    91: 120,
    92: 63,
    93: 64,
    94: 65,
    95: 66,
    96: 67,
    97: 68,
    98: 69,
    99: 70,
    100: 71,
    101: 72,
    102: 73,
    103: 74,
    104: 75,
    105: 76,
    106: 77,
    107: 78,
    108: 79,
    109: 80,
    110: 81,
    111: 82,
    112: 83,
    113: 84,
    114: 85,
    115: 86,
    116: 122,
    117: 114
}

def compute_dice_coefficient(y_true, y_pred):
    """
    Compute Dice coefficient: 2*|X∩Y|/(|X|+|Y|)
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def compute_class_dice(nac_path, ct_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Compute Dice coefficients for each class between NAC and CT tissue segmentations
    
    Args:
        nac_path: Path to the NAC tissue segmentation file
        ct_path: Path to the CT tissue segmentation file
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary containing Dice scores for each class and overall
    """
    # Check if files exist
    if not os.path.exists(nac_path):
        raise FileNotFoundError(f"NAC segmentation file not found: {nac_path}")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT segmentation file not found: {ct_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the segmentation files
    print("Loading segmentation files...")
    nac_ts_img = nib.load(nac_path)
    ct_ts_img = nib.load(ct_path)

    nac_ts_data = nac_ts_img.get_fdata()
    ct_ts_data = ct_ts_img.get_fdata()

    # Find unique class labels in both segmentations
    nac_unique_labels = np.unique(nac_ts_data).astype(int)
    ct_unique_labels = np.unique(ct_ts_data).astype(int)

    print(f"Unique labels in NAC TS: {nac_unique_labels}")
    print(f"Unique labels in CT TS: {ct_unique_labels}")

    # Find common labels to compute Dice for
    common_labels = np.intersect1d(nac_unique_labels, ct_unique_labels)
    all_labels = np.union1d(nac_unique_labels, ct_unique_labels)

    print(f"Common labels: {common_labels}")
    print(f"All labels: {all_labels}")

    # Store results
    results = {}
    
    # Compute Dice coefficient for each class
    print("\nDice coefficients for each class:")
    for label in all_labels:
        if label == 0:  # Skip background
            continue
            
        # Create binary masks for this class
        nac_mask = (nac_ts_data == label).astype(int)
        ct_mask = (ct_ts_data == label).astype(int)
        
        # Compute Dice
        if label in common_labels:
            dice_score = compute_dice_coefficient(nac_mask, ct_mask)
            print(f"Class {label}: {dice_score:.4f}")
            results[f"class_{label}"] = dice_score
        else:
            if label in nac_unique_labels:
                print(f"Class {label}: Only in NAC segmentation")
                results[f"class_{label}"] = "NAC only"
            else:
                print(f"Class {label}: Only in CT segmentation")
                results[f"class_{label}"] = "CT only"

    # Compute overall Dice (considering all non-zero labels as foreground)
    nac_foreground = (nac_ts_data > 0).astype(int)
    ct_foreground = (ct_ts_data > 0).astype(int)
    overall_dice = compute_dice_coefficient(nac_foreground, ct_foreground)
    print(f"\nOverall Dice (all foreground vs background): {overall_dice:.4f}")
    results["overall"] = overall_dice
    
    # Save results to a text file
    results_path = os.path.join(output_dir, "dice_results.txt")
    with open(results_path, 'w') as f:
        f.write("Dice Coefficient Results\n")
        f.write("=======================\n\n")
        f.write(f"NAC file: {nac_path}\n")
        f.write(f"CT file: {ct_path}\n\n")
        f.write("Class-wise Dice coefficients:\n")
        for label in all_labels:
            if label == 0:
                continue
            if f"class_{label}" in results:
                f.write(f"Class {label}: {results[f'class_{label}']}\n")
        f.write(f"\nOverall Dice: {overall_dice:.4f}\n")
    
    print(f"Dice results saved to {results_path}")
    return results

def convert_ts_to_maisi(ts_path, body_contour_pet_path, body_contour_ct_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Convert tissue segmentation (TS) labels to MAISI labels and add body contour
    
    Args:
        ts_path: Path to the TS segmentation file
        body_contour_pet_path: Path to the PET body contour file
        body_contour_ct_path: Path to the CT body contour file
        output_dir: Directory to save output files
        
    Returns:
        tuple: Paths to the saved MAISI segmentation files with PET and CT body contours
    """
    # Check if files exist
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"TS segmentation file not found: {ts_path}")
    if not os.path.exists(body_contour_pet_path):
        raise FileNotFoundError(f"PET body contour file not found: {body_contour_pet_path}")
    if not os.path.exists(body_contour_ct_path):
        raise FileNotFoundError(f"CT body contour file not found: {body_contour_ct_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the segmentation files
    print("Loading files for TS to MAISI conversion...")
    ts_img = nib.load(ts_path)
    body_contour_pet_img = nib.load(body_contour_pet_path)
    body_contour_ct_img = nib.load(body_contour_ct_path)
    
    ts_data = ts_img.get_fdata()
    body_contour_pet_data = body_contour_pet_img.get_fdata()
    body_contour_ct_data = body_contour_ct_img.get_fdata()
    
    # Create MAISI label maps (initialize with zeros)
    maisi_pet_data = np.zeros_like(ts_data)
    maisi_ct_data = np.zeros_like(ts_data)
    
    # Map TS labels to MAISI labels
    print("Converting TS labels to MAISI labels...")
    unique_labels = np.unique(ts_data).astype(int)
    print(f"Unique TS labels found: {unique_labels}")
    
    for ts_label in unique_labels:
        if ts_label == 0:  # Skip background
            continue
            
        if ts_label in T2M_mapping:
            maisi_label = T2M_mapping[ts_label]
            mask = (ts_data == ts_label)
            maisi_pet_data[mask] = maisi_label
            maisi_ct_data[mask] = maisi_label
        else:
            print(f"Warning: TS label {ts_label} not found in mapping dictionary")
    
    # Add body contour (class 200)
    print("Adding body contour (class 200)...")
    maisi_pet_data[body_contour_pet_data > 0] = 200
    maisi_ct_data[body_contour_ct_data > 0] = 200
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(ts_path))[0]
    if base_filename.endswith('.nii'):  # Handle .nii.gz case
        base_filename = os.path.splitext(base_filename)[0]
    
    # Save MAISI segmentation files
    maisi_pet_path = os.path.join(output_dir, f"{base_filename}_MAISI_bcP.nii.gz")
    maisi_ct_path = os.path.join(output_dir, f"{base_filename}_MAISI_bcC.nii.gz")
    
    print(f"Saving MAISI segmentation with PET body contour to: {maisi_pet_path}")
    maisi_pet_img = nib.Nifti1Image(maisi_pet_data, ts_img.affine, ts_img.header)
    nib.save(maisi_pet_img, maisi_pet_path)
    
    print(f"Saving MAISI segmentation with CT body contour to: {maisi_ct_path}")
    maisi_ct_img = nib.Nifti1Image(maisi_ct_data, ts_img.affine, ts_img.header)
    nib.save(maisi_ct_img, maisi_ct_path)
    
    # Count and report unique MAISI labels
    maisi_pet_unique = np.unique(maisi_pet_data).astype(int)
    maisi_ct_unique = np.unique(maisi_ct_data).astype(int)
    print(f"Unique MAISI labels in PET body contour version: {maisi_pet_unique}")
    print(f"Unique MAISI labels in CT body contour version: {maisi_ct_unique}")
    
    # Save mapping information to a text file
    mapping_path = os.path.join(output_dir, "ts_to_maisi_mapping.txt")
    with open(mapping_path, 'w') as f:
        f.write("TS to MAISI Label Mapping\n")
        f.write("========================\n\n")
        f.write(f"TS file: {ts_path}\n")
        f.write(f"PET body contour: {body_contour_pet_path}\n")
        f.write(f"CT body contour: {body_contour_ct_path}\n\n")
        f.write("Mapping used:\n")
        f.write("TS Label -> MAISI Label\n")
        for ts_label in sorted(T2M_mapping.keys()):
            f.write(f"{ts_label} -> {T2M_mapping[ts_label]}\n")
        f.write("\nBody contour -> 200\n")
        f.write(f"\nOutput files:\n{maisi_pet_path}\n{maisi_ct_path}\n")
    
    print(f"Mapping information saved to {mapping_path}")
    return maisi_pet_path, maisi_ct_path

def main():
    parser = argparse.ArgumentParser(description='Process NAC and CT tissue segmentations')
    parser.add_argument('--compute_dice', action='store_true', help='Compute Dice coefficients')
    parser.add_argument('--convert_ts_to_maisi', action='store_true', help='Convert TS labels to MAISI labels')
    parser.add_argument('--nac_path', type=str, default=NAC_TS_label_path, help='Path to NAC tissue segmentation file')
    parser.add_argument('--ct_path', type=str, default=CT_TS_label_path, help='Path to CT tissue segmentation file')
    parser.add_argument('--pet_contour_path', type=str, default=NAC_TS_body_contour_PET_path, help='Path to PET body contour file')
    parser.add_argument('--ct_contour_path', type=str, default=NAC_TS_body_contour_CT_path, help='Path to CT body contour file')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")
    
    if args.compute_dice:
        dice_results = compute_class_dice(args.nac_path, args.ct_path, args.output_dir)
        print("Dice computation completed.")
    
    if args.convert_ts_to_maisi:
        maisi_pet_path, maisi_ct_path = convert_ts_to_maisi(
            args.nac_path, 
            args.pet_contour_path, 
            args.ct_contour_path,
            args.output_dir
        )
        print(f"TS to MAISI conversion completed. Files saved to {args.output_dir}")
    
    if not (args.compute_dice or args.convert_ts_to_maisi):
        print("No action specified. Use --compute_dice or --convert_ts_to_maisi to activate functions.")
        print("Example usage:")
        print(f"  python {os.path.basename(__file__)} --compute_dice --convert_ts_to_maisi")

if __name__ == "__main__":
    main()

