import sys
import os
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from TS_NAC.map_to_binary import class_map_5_parts

def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    print("Creating dataset.json...")
    out_base = Path(os.environ['nnUNet_raw']) / foldername

    json_dict = {}
    json_dict['name'] = "TS_NAC"
    json_dict['description'] = "NAC to AC conversion using TotalSegmentator"
    json_dict['reference'] = ""
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "1.0"
    json_dict['channel_names'] = {"0": "NAC"}
    json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername
    output_folder_pkl.mkdir(exist_ok=True)

    # Create splits format expected by nnUNet
    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


def combine_labels(ref_img, file_out, masks):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx+1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


def extract_selected_labels(label_file, output_file, class_map):
    """
    Create a new label file containing only the selected organ classes
    
    Args:
        label_file: Path to the full label file
        output_file: Path to save the new label file with selected classes only
        class_map: Dictionary mapping label values to organ names (e.g., {1: "spleen", 2: "kidney_right", ...})
    """
    img = nib.load(label_file)
    data = img.get_fdata()
    
    # Create empty array for new labels
    new_data = np.zeros_like(data)
    
    # Map of original label values to new consecutive values starting from 1
    new_label_mapping = {orig_val: idx+1 for idx, orig_val in enumerate(class_map.keys())}
    
    # Copy only the selected organ classes with new label values
    for orig_val, new_val in new_label_mapping.items():
        new_data[data == orig_val] = new_val
    
    # Save the new label file
    new_img = nib.Nifti1Image(new_data.astype(np.uint8), img.affine)
    nib.save(new_img, output_file)


if __name__ == "__main__":
    """
    Convert the TS_NAC dataset to nnUNet format and generate dataset.json and splits_final.json
    
    Usage:
    python TSNAC_02_TS_dataset_to_nnUNet.py <dataset_path> <nnunet_dataset_path> <class_map_name>
    """
    dataset_path = Path(sys.argv[1])  # directory containing all the subjects
    nnunet_path = Path(sys.argv[2])  # directory of the new nnunet dataset
    class_map_name = sys.argv[3]  # which class map to use from map_to_binary

    split_json = dataset_path / "TS_NAC_split_cv0.json"
    
    # Get the class map from map_to_binary based on the provided name
    class_map = class_map_5_parts[class_map_name]

    (nnunet_path).mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    # Load split information
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    
    subjects_train = split_data['train']
    subjects_val = split_data['val']
    subjects_test = split_data['test']

    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        
        # Create new label file with selected organs only
        extract_selected_labels(
            subject_path / "label.nii.gz",
            nnunet_path / "labelsTr" / f"{subject}.nii.gz",
            class_map
        )

    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        
        # Create new label file with selected organs only
        extract_selected_labels(
            subject_path / "label.nii.gz",
            nnunet_path / "labelsTs" / f"{subject}.nii.gz",
            class_map
        )

    # Use the original label values (keys) as labels for dataset.json
    labels = list(class_map.keys())
    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, labels)


