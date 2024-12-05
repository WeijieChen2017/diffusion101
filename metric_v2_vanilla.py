root_dir = "projects/v1_vanilla_pet_cond"
eval_folds = [
    "test_results_ddim_batch_18_step_50",
    "test_results_ddim_500",
    "test_results_ddpm_batch_18",
]
mask_folder = "James_data_v3/mask"

import os
import numpy as np
import nibabel as nib
from collections import defaultdict

# Group slices by case name
def group_slices_by_case(fold_path):
    case_slices = defaultdict(list)
    for filename in os.listdir(fold_path):
        if filename.endswith('.npz'):
            # Extract case name (E4XXX) from filename
            case_name = filename[filename.find('E4'):filename.find('E4')+5]
            slice_path = os.path.join(fold_path, filename)
            case_slices[case_name].append(slice_path)
    return case_slices

# Process each evaluation fold
def process_eval_folds(root_dir, eval_folds, mask_folder):
    for fold in eval_folds:
        fold_path = os.path.join(root_dir, fold)
        if not os.path.exists(fold_path):
            print(f"Skipping non-existent fold: {fold_path}")
            continue
            
        # Group slices by case
        case_slices = group_slices_by_case(fold_path)
        
        # Process each case
        for case_name, slice_paths in case_slices.items():
            # Update mask path to match correct pattern
            mask_path = os.path.join(mask_folder, f"mask_body_contour_{case_name}.nii.gz")
            if not os.path.exists(mask_path):
                print(f"Mask not found for case {case_name} at {mask_path}")
                continue
                
            # Rest of the function remains the same
            mask_nifti = nib.load(mask_path)
            header = mask_nifti.header
            affine = mask_nifti.affine
            
            # Sort slices by slice number
            slice_paths.sort(key=lambda x: int(x.split('slice_')[-1].split('.')[0]))
            
            # Load and stack all slices
            pet_volume = []
            ct_volume = []
            pred_ct_volume = []
            
            for slice_path in slice_paths:
                data = np.load(slice_path)
                pet_volume.append(data['PET'])
                ct_volume.append(data['CT'])
                pred_ct_volume.append(data['Pred_CT'])
            
            # Convert to numpy arrays
            pet_volume = np.stack(pet_volume, axis=0)
            ct_volume = np.stack(ct_volume, axis=0)
            pred_ct_volume = np.stack(pred_ct_volume, axis=0)

            # Show the numpy shape
            print(f"pet shape {pet_volume.shape}, ct shape {ct_volume.shape}, pred ct shape {pred_ct_volume.shape}")

            
            # Create and save NIFTI files
            output_dir = os.path.join(root_dir, f"{fold}_nifti")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save PET volume
            pet_nifti = nib.Nifti1Image(pet_volume, affine, header)
            nib.save(pet_nifti, os.path.join(output_dir, f"PET_{case_name}.nii.gz"))
            
            # Save ground truth CT volume
            ct_nifti = nib.Nifti1Image(ct_volume, affine, header)
            nib.save(ct_nifti, os.path.join(output_dir, f"CT_{case_name}.nii.gz"))
            
            # Save predicted CT volume
            pred_ct_nifti = nib.Nifti1Image(pred_ct_volume, affine, header)
            nib.save(pred_ct_nifti, os.path.join(output_dir, f"Pred_CT_{case_name}.nii.gz"))
            
            print(f"Processed case {case_name} in fold {fold}")

# Main execution
if __name__ == "__main__":
    root_dir = "projects/v1_vanilla_pet_cond"
    eval_folds = [
        "test_results_ddim_batch_18_step_50",
        "test_results_ddim_500",
        "test_results_ddpm_batch_18",
    ]
    mask_folder = "James_data_v3/mask"
    
    process_eval_folds(root_dir, eval_folds, mask_folder)

