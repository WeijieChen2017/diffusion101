import os
import sys
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
from tqdm import tqdm

# Fixed paths
INPUT_DIR = "/local/diffusion101/maisi/TS_NAC"
MASK_DIR = "/local/diffusion101/maisi/NAC_CTAC_Spacing15"
DEFAULT_OUTPUT_DIR = "/local/diffusion101/maisi/BodyContour_Dataset"

# Predefined dataset split
dataset_split = {
    "train": [
        "E4242", "E4245", "E4246", "E4247", "E4248", "E4250", "E4252", "E4259", "E4260", "E4261",
        "E4273", "E4275", "E4276", "E4280", "E4282", "E4283", "E4284", "E4288", "E4290", "E4292",
        "E4297", "E4298", "E4299", "E4300", "E4301", "E4302", "E4306", "E4307", "E4308", "E4309",
        "E4310", "E4312"
    ],
    "val": [
        "E4313", "E4317", "E4318", "E4324", "E4325", "E4328", "E4332", "E4335", "E4336", "E4337",
        "E4338"
    ],
    "test": [
        "E4055", "E4058", "E4061", "E4066", "E4068", "E4069", "E4073", "E4074", "E4077", "E4078",
        "E4079", "E4081", "E4084", "E4091", "E4092", "E4094", "E4096", "E4098", "E4099", "E4103",
        "E4105", "E4106", "E4114", "E4115", "E4118", "E4120", "E4124", "E4125", "E4128", "E4129",
        "E4130", "E4131", "E4134", "E4137", "E4138", "E4139"
    ]
}

def prepare_dataset_for_nnunet(output_dir=DEFAULT_OUTPUT_DIR):
    """
    Prepare dataset for nnUNet training using CT images from subject folders and body contour masks
    
    Args:
        output_dir: Directory to save nnUNet dataset
                   (default: /local/diffusion101/maisi/BodyContour_Dataset)
    """
    input_dir = Path(INPUT_DIR)
    output_dir = Path(output_dir)
    mask_dir = Path(MASK_DIR)
    
    print(f"Input directory (fixed): {input_dir}")
    print(f"Mask directory (fixed): {mask_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create nnUNet directory structure
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTs").mkdir(parents=True, exist_ok=True)
    
    # Use predefined dataset split
    subjects_train = dataset_split["train"]
    subjects_val = dataset_split["val"]
    subjects_test = dataset_split["test"]
    
    print(f"Training subjects: {len(subjects_train)}")
    print(f"Validation subjects: {len(subjects_val)}")
    print(f"Test subjects: {len(subjects_test)}")
    
    # Process training and validation data
    for subject in tqdm(subjects_train + subjects_val, desc="Processing training and validation data"):
        # CT image path from subject folder
        subject_dir = input_dir / subject
        ct_path = subject_dir / "ct.nii.gz"
        
        # Body contour mask path from mask directory
        mask_path = mask_dir / f"CTAC_{subject}_TS_body.nii.gz"
        
        # Check if files exist
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        if not ct_path.exists():
            print(f"Warning: CT file not found: {ct_path}")
            continue
        if not mask_path.exists():
            print(f"Warning: Body contour mask file not found: {mask_path}")
            continue
        
        # Copy CT image to nnUNet imagesTr directory
        shutil.copy(ct_path, output_dir / "imagesTr" / f"{subject}_0000.nii.gz")
        
        # Load body contour mask and ensure it's binary (0 and 1)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        binary_mask = (mask_data > 0).astype(np.uint8)
        
        # Save binary mask to nnUNet labelsTr directory
        binary_mask_img = nib.Nifti1Image(binary_mask, mask_img.affine, mask_img.header)
        nib.save(binary_mask_img, output_dir / "labelsTr" / f"{subject}.nii.gz")
    
    # Process test data
    for subject in tqdm(subjects_test, desc="Processing test data"):
        # CT image path from subject folder
        subject_dir = input_dir / subject
        ct_path = subject_dir / "ct.nii.gz"
        
        # Body contour mask path from mask directory
        mask_path = mask_dir / f"CTAC_{subject}_TS_body.nii.gz"
        
        # Check if files exist
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        if not ct_path.exists():
            print(f"Warning: CT file not found: {ct_path}")
            continue
        if not mask_path.exists():
            print(f"Warning: Body contour mask file not found: {mask_path}")
            continue
        
        # Copy CT image to nnUNet imagesTs directory
        shutil.copy(ct_path, output_dir / "imagesTs" / f"{subject}_0000.nii.gz")
        
        # Load body contour mask and ensure it's binary (0 and 1)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        binary_mask = (mask_data > 0).astype(np.uint8)
        
        # Save binary mask to nnUNet labelsTs directory
        binary_mask_img = nib.Nifti1Image(binary_mask, mask_img.affine, mask_img.header)
        nib.save(binary_mask_img, output_dir / "labelsTs" / f"{subject}.nii.gz")
    
    # Create dataset.json
    create_dataset_json(output_dir, subjects_train, subjects_val)
    
    # Create splits_final.json
    create_splits_json(output_dir, subjects_train, subjects_val)
    
    print("Dataset preparation complete!")

def create_dataset_json(output_dir, subjects_train, subjects_val):
    """
    Create dataset.json file for nnUNet
    
    Args:
        output_dir: Directory to save dataset.json
        subjects_train: List of training subjects
        subjects_val: List of validation subjects
    """
    output_dir = Path(output_dir)
    
    json_dict = {}
    json_dict['name'] = "BodyContour"
    json_dict['description'] = "Body contour segmentation from CT images"
    json_dict['reference'] = ""
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "1.0"
    json_dict['channel_names'] = {"0": "CT"}
    json_dict['labels'] = {"0": "background", "1": "body"}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'
    
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(json_dict, f, sort_keys=False, indent=4)
    
    print(f"Created dataset.json at {output_dir / 'dataset.json'}")

def create_splits_json(output_dir, subjects_train, subjects_val):
    """
    Create splits_final.json file for nnUNet
    
    Args:
        output_dir: Directory to save splits_final.json
        subjects_train: List of training subjects
        subjects_val: List of validation subjects
    """
    output_dir = Path(output_dir)
    
    # Create nnUNet preprocessed directory
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    
    # Create splits format expected by nnUNet
    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })
    
    with open(preprocessed_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, sort_keys=False, indent=4)
    
    print(f"Created splits_final.json at {preprocessed_dir / 'splits_final.json'}")

if __name__ == "__main__":
    """
    Prepare dataset for nnUNet training using CT images from subject folders and body contour masks
    
    Usage:
    python TSNAC_03_body_seg.py [<output_dir>]
    
    Args:
        output_dir: Directory to save nnUNet dataset
                   (default: /local/diffusion101/maisi/BodyContour_Dataset)
    """
    if len(sys.argv) == 1:
        # Use default output path
        prepare_dataset_for_nnunet()
    elif len(sys.argv) == 2:
        # Use provided output path
        prepare_dataset_for_nnunet(sys.argv[1])
    else:
        print("Usage: python TSNAC_03_body_seg.py [<output_dir>]")
        print(f"Input directory (fixed): {INPUT_DIR}")
        print(f"Mask directory (fixed): {MASK_DIR}")
        print(f"Default output directory: {DEFAULT_OUTPUT_DIR}")
        sys.exit(1)
