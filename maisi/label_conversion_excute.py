import os
import nibabel as nib
import numpy as np
from pathlib import Path

TSv2_labels = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

MAISI_labels = {
    "liver": 1,
    "dummy1": 2,
    "spleen": 3,
    "pancreas": 4,
    "right kidney": 5,
    "aorta": 6,
    "inferior vena cava": 7,
    "right adrenal gland": 8,
    "left adrenal gland": 9,
    "gallbladder": 10,
    "esophagus": 11,
    "stomach": 12,
    "duodenum": 13,
    "left kidney": 14,
    "bladder": 15,
    "dummy2": 16,
    "portal vein and splenic vein": 17,
    "dummy3": 18,
    "small bowel": 19,
    "dummy4": 20,
    "dummy5": 21,
    "brain": 22,
    "lung tumor": 23,
    "pancreatic tumor": 24,
    "hepatic vessel": 25,
    "hepatic tumor": 26,
    "colon cancer primaries": 27,
    "left lung upper lobe": 28,
    "left lung lower lobe": 29,
    "right lung upper lobe": 30,
    "right lung middle lobe": 31,
    "right lung lower lobe": 32,
    "vertebrae L5": 33,
    "vertebrae L4": 34,
    "vertebrae L3": 35,
    "vertebrae L2": 36,
    "vertebrae L1": 37,
    "vertebrae T12": 38,
    "vertebrae T11": 39,
    "vertebrae T10": 40,
    "vertebrae T9": 41,
    "vertebrae T8": 42,
    "vertebrae T7": 43,
    "vertebrae T6": 44,
    "vertebrae T5": 45,
    "vertebrae T4": 46,
    "vertebrae T3": 47,
    "vertebrae T2": 48,
    "vertebrae T1": 49,
    "vertebrae C7": 50,
    "vertebrae C6": 51,
    "vertebrae C5": 52,
    "vertebrae C4": 53,
    "vertebrae C3": 54,
    "vertebrae C2": 55,
    "vertebrae C1": 56,
    "trachea": 57,
    "left iliac artery": 58,
    "right iliac artery": 59,
    "left iliac vena": 60,
    "right iliac vena": 61,
    "colon": 62,
    "left rib 1": 63,
    "left rib 2": 64,
    "left rib 3": 65,
    "left rib 4": 66,
    "left rib 5": 67,
    "left rib 6": 68,
    "left rib 7": 69,
    "left rib 8": 70,
    "left rib 9": 71,
    "left rib 10": 72,
    "left rib 11": 73,
    "left rib 12": 74,
    "right rib 1": 75,
    "right rib 2": 76,
    "right rib 3": 77,
    "right rib 4": 78,
    "right rib 5": 79,
    "right rib 6": 80,
    "right rib 7": 81,
    "right rib 8": 82,
    "right rib 9": 83,
    "right rib 10": 84,
    "right rib 11": 85,
    "right rib 12": 86,
    "left humerus": 87,
    "right humerus": 88,
    "left scapula": 89,
    "right scapula": 90,
    "left clavicula": 91,
    "right clavicula": 92,
    "left femur": 93,
    "right femur": 94,
    "left hip": 95,
    "right hip": 96,
    "sacrum": 97,
    "left gluteus maximus": 98,
    "right gluteus maximus": 99,
    "left gluteus medius": 100,
    "right gluteus medius": 101,
    "left gluteus minimus": 102,
    "right gluteus minimus": 103,
    "left autochthon": 104,
    "right autochthon": 105,
    "left iliopsoas": 106,
    "right iliopsoas": 107,
    "left atrial appendage": 108,
    "brachiocephalic trunk": 109,
    "left brachiocephalic vein": 110,
    "right brachiocephalic vein": 111,
    "left common carotid artery": 112,
    "right common carotid artery": 113,
    "costal cartilages": 114,
    "heart": 115,
    "left kidney cyst": 116,
    "right kidney cyst": 117,
    "prostate": 118,
    "pulmonary vein": 119,
    "skull": 120,
    "spinal cord": 121,
    "sternum": 122,
    "left subclavian artery": 123,
    "right subclavian artery": 124,
    "superior vena cava": 125,
    "thyroid gland": 126,
    "vertebrae S1": 127,
    "bone lesion": 128,
    "dummy6": 129,
    "dummy7": 130,
    "dummy8": 131,
    "airway": 132
}

class LabelConverter:
    def __init__(self, tsv2_labels, maisi_labels):
        self.tsv2_to_maisi = {}
        self.maisi_to_tsv2 = {}
        
        # Create mapping dictionaries
        for tsv2_idx, tsv2_name in tsv2_labels.items():
            # Handle special cases and naming differences
            converted_name = self._convert_tsv2_to_maisi_name(tsv2_name)
            
            # Find corresponding MAISI label
            for maisi_name, maisi_idx in maisi_labels.items():
                if converted_name == maisi_name:
                    self.tsv2_to_maisi[tsv2_idx] = maisi_idx
                    self.maisi_to_tsv2[maisi_idx] = tsv2_idx
                    break
    
    def _convert_tsv2_to_maisi_name(self, name):
        """Convert TSv2 naming convention to MAISI naming convention."""
        # Your existing name mapping dictionary here
        name_mapping = {
            'kidney_right': 'right kidney',
            'kidney_left': 'left kidney',
            'adrenal_gland_right': 'right adrenal gland',
            'adrenal_gland_left': 'left adrenal gland',
            'lung_upper_lobe_left': 'left lung upper lobe',
            'lung_lower_lobe_left': 'left lung lower lobe',
            'lung_upper_lobe_right': 'right lung upper lobe',
            'lung_middle_lobe_right': 'right lung middle lobe',
            'lung_lower_lobe_right': 'right lung lower lobe',
            'urinary_bladder': 'bladder',
            'portal_vein_and_splenic_vein': 'portal vein and splenic vein',
            'iliac_artery_left': 'left iliac artery',
            'iliac_artery_right': 'right iliac artery',
            'iliac_vena_left': 'left iliac vena',
            'iliac_vena_right': 'right iliac vena',
            'humerus_left': 'left humerus',
            'humerus_right': 'right humerus',
            'scapula_left': 'left scapula',
            'scapula_right': 'right scapula',
            'clavicula_left': 'left clavicula',
            'clavicula_right': 'right clavicula',
            'femur_left': 'left femur',
            'femur_right': 'right femur',
            'hip_left': 'left hip',
            'hip_right': 'right hip',
            'gluteus_maximus_left': 'left gluteus maximus',
            'gluteus_maximus_right': 'right gluteus maximus',
            'gluteus_medius_left': 'left gluteus medius',
            'gluteus_medius_right': 'right gluteus medius',
            'gluteus_minimus_left': 'left gluteus minimus',
            'gluteus_minimus_right': 'right gluteus minimus',
            'autochthon_left': 'left autochthon',
            'autochthon_right': 'right autochthon',
            'iliopsoas_left': 'left iliopsoas',
            'iliopsoas_right': 'right iliopsoas',
            'atrial_appendage_left': 'left atrial appendage',
            'brachiocephalic_vein_left': 'left brachiocephalic vein',
            'brachiocephalic_vein_right': 'right brachiocephalic vein',
            'common_carotid_artery_left': 'left common carotid artery',
            'common_carotid_artery_right': 'right common carotid artery',
            'kidney_cyst_left': 'left kidney cyst',
            'kidney_cyst_right': 'right kidney cyst',
            'subclavian_artery_left': 'left subclavian artery',
            'subclavian_artery_right': 'right subclavian artery',
            'rib_left_1': 'left rib 1',
            'rib_left_2': 'left rib 2',
            'rib_left_3': 'left rib 3',
            'rib_left_4': 'left rib 4',
            'rib_left_5': 'left rib 5',
            'rib_left_6': 'left rib 6',
            'rib_left_7': 'left rib 7',
            'rib_left_8': 'left rib 8',
            'rib_left_9': 'left rib 9',
            'rib_left_10': 'left rib 10',
            'rib_left_11': 'left rib 11',
            'rib_left_12': 'left rib 12',
            'rib_right_1': 'right rib 1',
            'rib_right_2': 'right rib 2',
            'rib_right_3': 'right rib 3',
            'rib_right_4': 'right rib 4',
            'rib_right_5': 'right rib 5',
            'rib_right_6': 'right rib 6',
            'rib_right_7': 'right rib 7',
            'rib_right_8': 'right rib 8',
            'rib_right_9': 'right rib 9',
            'rib_right_10': 'right rib 10',
            'rib_right_11': 'right rib 11',
            'rib_right_12': 'right rib 12'
        }
        return name_mapping.get(name, name)
    
    def tsv2_to_maisi_index(self, tsv2_idx):
        """Convert TSv2 index to MAISI index."""
        return self.tsv2_to_maisi.get(tsv2_idx, 0)  # Return 0 for background/unmapped labels

def convert_nifti_files(input_folder, converter):
    """
    Convert all NIfTI files in the input folder from TSv2 to MAISI labels.
    
    Args:
        input_folder (str): Path to the folder containing NIfTI files
        converter (LabelConverter): Initialized label converter instance
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
            output_data = np.zeros_like(data)
            
            # Convert unique labels
            unique_labels = np.unique(data)
            for label in unique_labels:
                if label != 0:  # Skip background
                    maisi_label = converter.tsv2_to_maisi_index(int(label))
                    output_data[data == label] = maisi_label
            
            # Create new NIfTI image with same header and affine
            output_img = nib.Nifti1Image(output_data, nifti_img.affine, nifti_img.header)
            
            # Generate output filename
            output_filename = file_path.stem.replace('.nii', '') + '_MAISI.nii'
            if file_path.suffix == '.gz':
                output_filename += '.gz'
            
            # Save converted file
            output_path = file_path.parent / output_filename
            nib.save(output_img, str(output_path))
            
            print(f"Saved converted file: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

def main():
    # Initialize the converter with your label dictionaries
    converter = LabelConverter(TSv2_labels, MAISI_labels)
    
    # Get input folder from user
    input_folder = input("Please enter the path to the folder containing NIfTI files: ").strip()
    
    # Validate folder exists
    if not os.path.isdir(input_folder):
        print("Error: Invalid folder path")
        return
    
    # Process files
    convert_nifti_files(input_folder, converter)
    print("Conversion complete!")

if __name__ == "__main__":
    main()