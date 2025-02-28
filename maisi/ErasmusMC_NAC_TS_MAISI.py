import nibabel as nib
import numpy as np
import argparse
import os

case_name_list = [
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

# Template paths - will be formatted with case_name
NAC_TS_label_template = "combined_predictions/{}_combined.nii.gz"
NAC_TS_body_contour_CT_template = "James_36/CT_mask/mask_body_contour_{}.nii.gz"
CT_TS_label_template = "NAC_CTAC_Spacing15/CTAC_{}_TS.nii.gz"

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

def convert_ts_to_maisi(ts_path, body_contour_ct_path, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Convert tissue segmentation (TS) labels to MAISI labels and add body contour
    
    Args:
        ts_path: Path to the TS segmentation file
        body_contour_ct_path: Path to the CT body contour file
        output_dir: Directory to save output files
        
    Returns:
        str: Path to the saved MAISI segmentation file with CT body contour
    """
    # Check if files exist
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"TS segmentation file not found: {ts_path}")
    if not os.path.exists(body_contour_ct_path):
        raise FileNotFoundError(f"CT body contour file not found: {body_contour_ct_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the segmentation files
    print("Loading files for TS to MAISI conversion...")
    ts_img = nib.load(ts_path)
    body_contour_ct_img = nib.load(body_contour_ct_path)
    
    ts_data = ts_img.get_fdata()
    body_contour_ct_data = body_contour_ct_img.get_fdata()
    
    # Create MAISI label map (initialize with zeros)
    maisi_ct_data = np.zeros_like(ts_data, dtype=np.int32)
    
    # Map TS labels to MAISI labels
    print("Converting TS labels to MAISI labels...")
    unique_labels = np.unique(ts_data).astype(int)
    print(f"Unique TS labels found: {unique_labels}")
    
    # First, filter all labels outside the body contour
    print("Filtering labels outside body contour...")
    filtered_ts_ct_data = np.zeros_like(ts_data, dtype=np.int32)
    
    # Only keep labels inside the body contour - ensure integer type
    filtered_ts_ct_data = np.where(body_contour_ct_data > 0, ts_data, 0).astype(np.int32)
    
    # Map TS labels to MAISI labels for the filtered data
    for ts_label in unique_labels:
        if ts_label == 0:  # Skip background
            continue
            
        if ts_label in T2M_mapping:
            maisi_label = T2M_mapping[ts_label]
            
            # Apply mapping for CT body contour version
            mask_ct = (filtered_ts_ct_data == ts_label)
            maisi_ct_data[mask_ct] = maisi_label
        else:
            print(f"Warning: TS label {ts_label} not found in mapping dictionary")
    
    # Add body contour (class 200) to areas that are in the body contour but don't have a tissue label
    print("Adding body contour (class 200)...")
    # For CT: where body contour exists but no label has been assigned
    maisi_ct_data = np.where((body_contour_ct_data > 0) & (maisi_ct_data == 0), 200, maisi_ct_data).astype(np.int32)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(ts_path))[0]
    if base_filename.endswith('.nii'):  # Handle .nii.gz case
        base_filename = os.path.splitext(base_filename)[0]
    
    # Save MAISI segmentation file
    maisi_ct_path = os.path.join(output_dir, f"{base_filename}_MAISI_bcC.nii.gz")
    
    print(f"Saving MAISI segmentation with CT body contour to: {maisi_ct_path}")
    maisi_ct_img = nib.Nifti1Image(maisi_ct_data.astype(np.int32), ts_img.affine, ts_img.header)
    nib.save(maisi_ct_img, maisi_ct_path)
    
    # Count and report unique MAISI labels
    maisi_ct_unique = np.unique(maisi_ct_data).astype(int)
    print(f"Unique MAISI labels in CT body contour version: {maisi_ct_unique}")
    
    # Save mapping information to a text file
    mapping_path = os.path.join(output_dir, f"{base_filename}_mapping.txt")
    with open(mapping_path, 'w') as f:
        f.write("TS to MAISI Label Mapping\n")
        f.write("========================\n\n")
        f.write(f"TS file: {ts_path}\n")
        f.write(f"CT body contour: {body_contour_ct_path}\n\n")
        f.write("Mapping used:\n")
        f.write("TS Label -> MAISI Label\n")
        for ts_label in sorted(T2M_mapping.keys()):
            f.write(f"{ts_label} -> {T2M_mapping[ts_label]}\n")
        f.write("\nBody contour (where no other label exists) -> 200\n")
        f.write(f"\nOutput file:\n{maisi_ct_path}\n")
    
    print(f"Mapping information saved to {mapping_path}")
    return maisi_ct_path

def process_all_cases(output_dir=DEFAULT_OUTPUT_DIR):
    """Process all cases in the case_name_list"""
    print(f"Processing {len(case_name_list)} cases...")
    
    results = []
    for case_name in case_name_list:
        print(f"\n{'='*50}")
        print(f"Processing case: {case_name}")
        print(f"{'='*50}")
        
        # Format the file paths with the current case name
        nac_ts_path = NAC_TS_label_template.format(case_name)
        ct_body_contour_path = NAC_TS_body_contour_CT_template.format(case_name)
        
        try:
            # Convert TS to MAISI for this case
            maisi_ct_path = convert_ts_to_maisi(
                nac_ts_path,
                ct_body_contour_path,
                output_dir
            )
            results.append((case_name, "Success", maisi_ct_path))
        except Exception as e:
            print(f"Error processing case {case_name}: {str(e)}")
            results.append((case_name, "Failed", str(e)))
    
    # Print summary
    print("\n\nProcessing Summary:")
    print("="*50)
    success_count = sum(1 for r in results if r[1] == "Success")
    print(f"Successfully processed {success_count} out of {len(case_name_list)} cases")
    
    for case_name, status, message in results:
        print(f"{case_name}: {status}")
        if status == "Failed":
            print(f"  - Error: {message}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("TS to MAISI Processing Summary\n")
        f.write("=============================\n\n")
        f.write(f"Successfully processed {success_count} out of {len(case_name_list)} cases\n\n")
        
        for case_name, status, message in results:
            f.write(f"{case_name}: {status}\n")
            if status == "Failed":
                f.write(f"  - Error: {message}\n")
    
    print(f"Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Process NAC and CT tissue segmentations')
    parser.add_argument('--convert_ts_to_maisi', action='store_true', help='Convert TS labels to MAISI labels')
    parser.add_argument('--process_all', action='store_true', help='Process all cases in the case_name_list')
    parser.add_argument('--case_name', type=str, help='Process a specific case by name')
    parser.add_argument('--nac_path', type=str, help='Path to NAC tissue segmentation file')
    parser.add_argument('--ct_contour_path', type=str, help='Path to CT body contour file')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")
    
    if args.process_all:
        process_all_cases(args.output_dir)
    elif args.case_name:
        # Format the file paths with the specified case name
        nac_ts_path = NAC_TS_label_template.format(args.case_name)
        ct_body_contour_path = NAC_TS_body_contour_CT_template.format(args.case_name)
        
        maisi_ct_path = convert_ts_to_maisi(
            nac_ts_path,
            ct_body_contour_path,
            args.output_dir
        )
        print(f"TS to MAISI conversion completed for case {args.case_name}. File saved to {maisi_ct_path}")
    elif args.convert_ts_to_maisi and args.nac_path and args.ct_contour_path:
        maisi_ct_path = convert_ts_to_maisi(
            args.nac_path, 
            args.ct_contour_path,
            args.output_dir
        )
        print(f"TS to MAISI conversion completed. File saved to {maisi_ct_path}")
    else:
        print("No action specified. Use one of the following options:")
        print("  --process_all to process all cases in the case_name_list")
        print("  --case_name to process a specific case")
        print("  --convert_ts_to_maisi with --nac_path and --ct_contour_path to process a custom file")
        print("\nExample usage:")
        print(f"  python {os.path.basename(__file__)} --process_all")
        print(f"  python {os.path.basename(__file__)} --case_name E4055")
        print(f"  python {os.path.basename(__file__)} --convert_ts_to_maisi --nac_path path/to/nac.nii.gz --ct_contour_path path/to/contour.nii.gz")

if __name__ == "__main__":
    main()

