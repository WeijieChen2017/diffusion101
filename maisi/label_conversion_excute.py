import os
import nibabel as nib
import numpy as np
from pathlib import Path

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

def convert_nifti_files(input_folder):
    """
    Convert all NIfTI files in the input folder from TSv2 to MAISI labels.
    
    Args:
        input_folder (str): Path to the folder containing NIfTI files
    """
    # Create output folder if it doesn't exist
    input_path = Path(input_folder)
    
    # Process all .nii.gz and .nii files
    nifti_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
    
    print(f"Found {len(nifti_files)} NIfTI files to process")
    
    for file_path in nifti_files:
        try:
            # Load the NIfTI file
            print(f"Processing {file_path.name}...")
            nifti_img = nib.load(str(file_path))
            
            # Get image data
            data = nifti_img.get_fdata()
            
            # Create output array
            output_data = np.zeros_like(data, dtype=np.int32)
            
            # Convert labels
            for tsv2_idx, maisi_idx in T2M_mapping.items():
                output_data[data == tsv2_idx] = maisi_idx
            
            # Create new NIfTI image with same header and affine
            output_img = nib.Nifti1Image(output_data, nifti_img.affine, nifti_img.header)
            
            # Generate output filename
            output_filename = file_path.stem.replace('.nii', '') + '_MAISI.nii.gz'
            # if file_path.suffix == '.gz':
            #     output_filename += '.gz'
            
            # Save converted file
            output_path = file_path.parent / output_filename
            nib.save(output_img, str(output_path))
            
            print(f"Saved converted file: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

def main():
    
    # Get input folder from user
    input_folder = input("Please enter the path to the folder containing NIfTI files: ").strip()
    
    # Validate folder exists
    if not os.path.isdir(input_folder):
        print("Error: Invalid folder path")
        return
    
    # Process files
    convert_nifti_files(input_folder)
    print("Conversion complete!")

if __name__ == "__main__":
    main()