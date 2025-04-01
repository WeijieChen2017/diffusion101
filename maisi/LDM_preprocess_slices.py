import os
import json
import numpy as np
from tqdm import tqdm
import nibabel as nib
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

@dataclass
class SliceConfig:
    min_val: float = -1024
    max_val: float = 1976
    output_dir: str = ""

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'LDM_preprocess.log')),
            logging.StreamHandler()
        ]
    )

def normalize_ct(ct_array: np.ndarray, min_val: float = -1024, max_val: float = 1976) -> np.ndarray:
    """
    Normalize CT values from [min_val, max_val] to [0, 1]
    """
    # Clip values to the specified range
    ct_array = np.clip(ct_array, min_val, max_val)
    # Normalize to [0, 1]
    ct_array = (ct_array - min_val) / (max_val - min_val)
    return ct_array

def load_nifti(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load NIfTI file and return array and metadata using nibabel
    """
    img = nib.load(file_path)
    array = img.get_fdata()
    metadata = {
        'affine': img.affine,
        'header': img.header,
        'shape': img.shape
    }
    return array, metadata

def get_adjacent_slices(array: np.ndarray, slice_idx: int) -> np.ndarray:
    """
    Get three adjacent slices and handle edge cases
    """
    num_slices = array.shape[0]
    
    # Handle edge cases
    if num_slices == 1:
        return np.stack([array[0], array[0], array[0]], axis=0)
    elif num_slices == 2:
        if slice_idx == 0:
            return np.stack([array[0], array[0], array[1]], axis=0)
        else:
            return np.stack([array[0], array[1], array[1]], axis=0)
    else:
        # Normal case: get previous, current, and next slice
        prev_idx = max(0, slice_idx - 1)
        next_idx = min(num_slices - 1, slice_idx + 1)
        return np.stack([array[prev_idx], array[slice_idx], array[next_idx]], axis=0)

def process_single_case(
    ct_path: str,
    sct_path: str,
    case_name: str,
    config: SliceConfig
) -> None:
    """
    Process a single case by:
    1. Loading CT and sCT volumes
    2. Normalizing values
    3. Creating 3-channel images from adjacent slices
    4. Saving the processed data
    """
    # Create output directories
    ct_slices_dir = os.path.join(config.output_dir, 'LDM_CT_slices', case_name)
    sct_slices_dir = os.path.join(config.output_dir, 'LDM_sCT_slices', case_name)
    os.makedirs(ct_slices_dir, exist_ok=True)
    os.makedirs(sct_slices_dir, exist_ok=True)

    # Load volumes
    ct_array, _ = load_nifti(ct_path)
    sct_array, _ = load_nifti(sct_path)

    # Normalize volumes
    ct_array = normalize_ct(ct_array, config.min_val, config.max_val)
    sct_array = normalize_ct(sct_array, config.min_val, config.max_val)

    # Process each slice
    num_slices = ct_array.shape[0]
    for slice_idx in range(num_slices):
        # Get adjacent slices for both CT and sCT
        ct_slices = get_adjacent_slices(ct_array, slice_idx)
        sct_slices = get_adjacent_slices(sct_array, slice_idx)

        # Save slices
        ct_slice_path = os.path.join(ct_slices_dir, f'LDM_slice_{slice_idx:03d}.npy')
        sct_slice_path = os.path.join(sct_slices_dir, f'LDM_slice_{slice_idx:03d}.npy')

        np.save(ct_slice_path, ct_slices)
        np.save(sct_slice_path, sct_slices)

def preprocess_all_cases(
    data_div_json: str,
    output_dir: str,
    min_val: float = -1024,
    max_val: float = 1976
) -> None:
    """
    Preprocess all cases from the data division JSON file
    """
    # Setup configuration
    config = SliceConfig(
        min_val=min_val,
        max_val=max_val,
        output_dir=output_dir
    )

    # Setup logging
    setup_logging(output_dir)

    # Load data division
    with open(data_div_json, 'r') as f:
        data_div = json.load(f)

    # Create a set of all unique case names
    all_cases = set()
    for fold in data_div.values():
        for split in fold.values():
            all_cases.update(split)
    
    logging.info(f"Total unique cases to process: {len(all_cases)}")

    # Process each case
    for case_name in tqdm(all_cases, desc="Processing all cases"):
        try:
            # Construct paths
            ct_path = f"LDM_adapter/data/CT/CTAC_{case_name}_cropped.nii.gz"
            sct_path = f"LDM_adapter/data/sCT/CTAC_{case_name}_TS_MAISI.nii.gz"

            # Process the case
            process_single_case(
                ct_path=ct_path,
                sct_path=sct_path,
                case_name=case_name,
                config=config
            )
            
            logging.info(f"Successfully processed case: {case_name}")
        except Exception as e:
            logging.error(f"Error processing case {case_name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    data_div_json = "LDM_adapter/folds.json"
    output_dir = "LDM_adapter/dataset/slices"
    
    preprocess_all_cases(
        data_div_json=data_div_json,
        output_dir=output_dir,
        min_val=-1024,
        max_val=1976
    ) 