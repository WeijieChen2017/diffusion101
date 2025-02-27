# part order
# class_map_part_organs
# class_map_part_vertebrae
# class_map_part_cardiac
# class_map_part_muscles
# class_map_part_ribs

import nibabel as nib
import numpy as np
import os
import json
from pathlib import Path

# Copy all necessary class maps directly
class_map_part_organs = {
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
}

class_map_part_vertebrae = {
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
    26: "vertebrae_C1",
}

class_map_part_cardiac = {
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
    18: "iliac_vena_right",
}

class_map_part_muscles = {
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
    23: "skull",
}

class_map_part_ribs = {
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
    26: "costal_cartilages",
}

# Define task paths
task_101 = "uuUNet/Dataset101_OR0/imagesTs_pred/"
task_102 = "uuUNet/Dataset102_VE0/imagesTs_pred/"
task_103 = "uuUNet/Dataset103_CA0/imagesTs_pred/"
task_104 = "uuUNet/Dataset104_MU0/imagesTs_pred/"
task_105 = "uuUNet/Dataset105_RI0/imagesTs_pred/"

# Define test case names
Ts_namelist = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
]

def combine_predictions(output_dir="combined_predictions"):
    """
    Combine predictions from different nnUNet tasks into a single segmentation.
    Each class will be assigned a unique ID based on its task and original class ID.
    Following the specified part order:
    1. class_map_part_organs (1-24)
    2. class_map_part_vertebrae (25-50)
    3. class_map_part_cardiac (51-68)
    4. class_map_part_muscles (69-91)
    5. class_map_part_ribs (92-117)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define task paths and class maps in the specified order
    tasks = [
        (task_101, class_map_part_organs),
        (task_102, class_map_part_vertebrae),
        (task_103, class_map_part_cardiac),
        (task_104, class_map_part_muscles),
        (task_105, class_map_part_ribs)
    ]
    
    # Calculate offsets for each part
    offsets = [0]  # First part starts at 0 offset
    for i in range(len(tasks) - 1):
        offsets.append(offsets[-1] + len(tasks[i][1]))
    
    # Create a combined class map with sequential indices
    combined_class_map = {}
    for i, (_, class_map) in enumerate(tasks):
        for original_id, organ_name in class_map.items():
            new_id = original_id + offsets[i]
            combined_class_map[new_id] = organ_name
    
    # Save the combined class map
    with open(os.path.join(output_dir, "combined_class_map.json"), "w") as f:
        json.dump(combined_class_map, f, indent=4)
    
    print(f"Starting to process {len(Ts_namelist)} test cases...")
    
    # Process each test case
    for idx, case_name in enumerate(Ts_namelist):
        print(f"Processing case [{idx+1}/{len(Ts_namelist)}]: {case_name}")
        
        # Get reference image from first task to determine dimensions
        ref_img_path = os.path.join(tasks[0][0], f"{case_name}.nii.gz")
        if not os.path.exists(ref_img_path):
            print(f"  ERROR: Reference image not found: {ref_img_path}")
            continue
        
        ref_img = nib.load(ref_img_path)
        combined_seg = np.zeros(ref_img.shape, dtype=np.uint16)
        
        # Process each task
        for task_idx, (task_path, class_map) in enumerate(tasks):
            task_name = os.path.basename(os.path.dirname(task_path.rstrip('/')))
            print(f"  Processing task {task_idx+1}/{len(tasks)}: {task_name}")
            
            img_path = os.path.join(task_path, f"{case_name}.nii.gz")
            if not os.path.exists(img_path):
                print(f"  WARNING: Image not found: {img_path}")
                continue
            
            # Load segmentation
            seg_img = nib.load(img_path)
            seg_data = seg_img.get_fdata().astype(np.int32)
            
            # Map classes to new indices with appropriate offset
            offset = offsets[task_idx]
            classes_found = 0
            for original_id in range(1, len(class_map) + 1):
                mask = seg_data == original_id
                if np.any(mask):
                    combined_seg[mask] = original_id + offset
                    classes_found += 1
            
            print(f"    Found {classes_found} classes in this task")
        
        # Save combined segmentation
        combined_nib = nib.Nifti1Image(combined_seg, ref_img.affine, ref_img.header)
        output_path = os.path.join(output_dir, f"{case_name}_combined.nii.gz")
        nib.save(combined_nib, output_path)
        print(f"  Saved combined prediction for {case_name}")
        print(f"  Completed case [{idx+1}/{len(Ts_namelist)}]")
    
    print(f"All {len(Ts_namelist)} cases processed successfully!")

if __name__ == "__main__":
    combine_predictions()

