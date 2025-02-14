import sys
import os
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# from TS_NAC.map_to_binary import class_map_5_parts

class_map_5_parts = {
    # 24 classes
    "class_map_part_organs": {
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
        24: "kidney_cyst_right"
    },

    # 26 classes
    "class_map_part_vertebrae": {
        1: "sacrum",
        2: "vertebrae_S1",
        3: "vertebrae_L5",
        4: "vertebrae_L4",
        5: "vertebrae_L3",
        6: "vertebrae_L2",
        7: "vertebrae_L1",
        8: "vertebrae_T12",
        9: "vertebrae_T11",
        10: "vertebrae_T10",
        11: "vertebrae_T9",
        12: "vertebrae_T8",
        13: "vertebrae_T7",
        14: "vertebrae_T6",
        15: "vertebrae_T5",
        16: "vertebrae_T4",
        17: "vertebrae_T3",
        18: "vertebrae_T2",
        19: "vertebrae_T1",
        20: "vertebrae_C7",
        21: "vertebrae_C6",
        22: "vertebrae_C5",
        23: "vertebrae_C4",
        24: "vertebrae_C3",
        25: "vertebrae_C2",
        26: "vertebrae_C1"
    },

    # 18
    "class_map_part_cardiac": {
        1: "heart",
        2: "aorta",
        3: "pulmonary_vein",
        4: "brachiocephalic_trunk",
        5: "subclavian_artery_right",
        6: "subclavian_artery_left",
        7: "common_carotid_artery_right",
        8: "common_carotid_artery_left",
        9: "brachiocephalic_vein_left",
        10: "brachiocephalic_vein_right",
        11: "atrial_appendage_left",
        12: "superior_vena_cava",
        13: "inferior_vena_cava",
        14: "portal_vein_and_splenic_vein",
        15: "iliac_artery_left",
        16: "iliac_artery_right",
        17: "iliac_vena_left",
        18: "iliac_vena_right"
    },

    # 23
    "class_map_part_muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "spinal_cord",
        12: "gluteus_maximus_left",
        13: "gluteus_maximus_right",
        14: "gluteus_medius_left",
        15: "gluteus_medius_right",
        16: "gluteus_minimus_left",
        17: "gluteus_minimus_right",
        18: "autochthon_left",
        19: "autochthon_right",
        20: "iliopsoas_left",
        21: "iliopsoas_right",
        22: "brain",
        23: "skull"
    },

    # 26 classes
    # 12. ribs start from vertebrae T12
    # Small subset of population (roughly 8%) have 13. rib below 12. rib
    #  (would start from L1 then)
    #  -> this has label rib_12
    # Even smaller subset (roughly 1%) has extra rib above 1. rib   ("Halsrippe")
    #  (the extra rib would start from C7)
    #  -> this has label rib_1
    #
    # Quite often only 11 ribs (12. ribs probably so small that not found). Those
    # cases often wrongly segmented.
    "class_map_part_ribs": {
        1: "rib_left_1",
        2: "rib_left_2",
        3: "rib_left_3",
        4: "rib_left_4",
        5: "rib_left_5",
        6: "rib_left_6",
        7: "rib_left_7",
        8: "rib_left_8",
        9: "rib_left_9",
        10: "rib_left_10",
        11: "rib_left_11",
        12: "rib_left_12",
        13: "rib_right_1",
        14: "rib_right_2",
        15: "rib_right_3",
        16: "rib_right_4",
        17: "rib_right_5",
        18: "rib_right_6",
        19: "rib_right_7",
        20: "rib_right_8",
        21: "rib_right_9",
        22: "rib_right_10",
        23: "rib_right_11",
        24: "rib_right_12",
        25: "sternum",
        26: "costal_cartilages"
    },   # "test": class_map["test"]
}

offset_labels = [
    0,
    len(class_map_5_parts["class_map_part_organs"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"])+len(class_map_5_parts["class_map_part_cardiac"]),
    len(class_map_5_parts["class_map_part_organs"])+len(class_map_5_parts["class_map_part_vertebrae"])+len(class_map_5_parts["class_map_part_cardiac"])+len(class_map_5_parts["class_map_part_muscles"]),
]

offset_labels = [
    0,
    24,
    24+26,
    24+26+28,
    24+26+28+23,
]


def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, labels):
    print("Creating dataset.json...")
    out_base = Path(os.environ['nnUNet_raw']) / foldername
    out_base.mkdir(parents=True, exist_ok=True)

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
    print(out_base / "dataset.json is saved.")

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / foldername
    output_folder_pkl.mkdir(parents=True, exist_ok=True)

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
    print(output_folder_pkl / "splits_final.json is saved.")

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


def get_label_range(class_map_name):
    """
    Get the label range for the given class map part
    """
    if class_map_name == "class_map_part_organs":
        return range(1, len(class_map_5_parts["class_map_part_organs"]) + 1)
    elif class_map_name == "class_map_part_vertebrae":
        start = len(class_map_5_parts["class_map_part_organs"]) + 1
        end = start + len(class_map_5_parts["class_map_part_vertebrae"])
        return range(start, end + 1)
    elif class_map_name == "class_map_part_cardiac":
        start = len(class_map_5_parts["class_map_part_organs"]) + len(class_map_5_parts["class_map_part_vertebrae"]) + 1
        end = start + len(class_map_5_parts["class_map_part_cardiac"])
        return range(start, end + 1)
    elif class_map_name == "class_map_part_muscles":
        start = len(class_map_5_parts["class_map_part_organs"]) + len(class_map_5_parts["class_map_part_vertebrae"]) + len(class_map_5_parts["class_map_part_cardiac"]) + 1
        end = start + len(class_map_5_parts["class_map_part_muscles"])
        return range(start, end + 1)
    elif class_map_name == "class_map_part_ribs":
        start = len(class_map_5_parts["class_map_part_organs"]) + len(class_map_5_parts["class_map_part_vertebrae"]) + len(class_map_5_parts["class_map_part_cardiac"]) + len(class_map_5_parts["class_map_part_muscles"]) + 1
        end = start + len(class_map_5_parts["class_map_part_ribs"])
        return range(start, end + 1)
    return range(0)

def get_label_mapping(class_map_name, class_map):
    """
    Get and verify the label mapping for the given class map part
    """
    # Get the label range for this class map part
    label_range = get_label_range(class_map_name)
    
    # Map of original label values (with offset) to new consecutive values starting from 1
    new_label_mapping = {orig_val: idx + 1 for idx, orig_val in enumerate(label_range)}
    
    # Print mapping for verification
    print(f"\nLabel mapping for {class_map_name}:")
    for orig_val, new_val in new_label_mapping.items():
        organ_name = class_map[new_val]  # Use new_val as key for class_map since it matches the original class numbering
        print(f"Original label {orig_val} ({organ_name}) -> New label {new_val}")
    
    # Ask for confirmation before proceeding
    proceed = input("\nDoes the mapping look correct? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Aborting operation...")
        sys.exit(1)
    
    return new_label_mapping

def extract_selected_labels(label_file, output_file, label_mapping):
    """
    Create a new label file containing only the selected organ classes
    Maps from offset labels to consecutive numbers starting from 1
    """
    img = nib.load(label_file)
    data = img.get_fdata()
    
    # Create empty array for new labels
    new_data = np.zeros_like(data)
    
    # Copy only the selected organ classes with new label values
    for orig_val, new_val in label_mapping.items():
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

    # Set environment variables
    split_json = dataset_path / "TS_NAC_split_cv0.json"
    
    # Get the class map from map_to_binary based on the provided name
    class_map = class_map_5_parts[class_map_name]

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

    # Get and verify label mapping once
    label_mapping = get_label_mapping(class_map_name, class_map)

    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTr" / f"{subject}_0000.nii.gz")
        
        # Create new label file with selected organs only
        extract_selected_labels(
            subject_path / "label.nii.gz",
            nnunet_path / "labelsTr" / f"{subject}.nii.gz",
            label_mapping
        )

    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        shutil.copy(subject_path / "ct.nii.gz", nnunet_path / "imagesTs" / f"{subject}_0000.nii.gz")
        
        # Create new label file with selected organs only
        extract_selected_labels(
            subject_path / "label.nii.gz",
            nnunet_path / "labelsTs" / f"{subject}.nii.gz",
            label_mapping
        )

    # Use the original label values (keys) as labels for dataset.json
    labels = list(class_map.keys())
    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, labels)


