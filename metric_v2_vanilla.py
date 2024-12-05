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
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def compute_metrics(ct_volume, pred_ct_volume):
    """Compute MAE, SSIM, and PSNR metrics."""
    # MAE calculation (multiply by 4000 as per original code)
    mae = np.mean(np.abs(ct_volume - pred_ct_volume)) * 4000
    
    # Compute SSIM and PSNR for each slice and average
    num_slices = ct_volume.shape[-1]
    ssim_scores = []
    psnr_scores = []
    
    for slice_idx in range(num_slices):
        ct_slice = ct_volume[..., slice_idx]
        pred_slice = pred_ct_volume[..., slice_idx]
        
        ssim_score = ssim(ct_slice, pred_slice, data_range=1.0)
        psnr_score = psnr(ct_slice, pred_slice, data_range=1.0)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    return {
        'MAE': mae,
        'SSIM': np.mean(ssim_scores),
        'PSNR': np.mean(psnr_scores)
    }

# Process each evaluation fold
def process_eval_folds(root_dir, eval_folds, mask_folder):
    results = []
    
    for fold in eval_folds:
        fold_path = os.path.join(root_dir, fold)
        if not os.path.exists(fold_path):
            print(f"Skipping non-existent fold: {fold_path}")
            continue
            
        # Group slices by case
        case_slices = group_slices_by_case(fold_path)
        
        # Process each case
        for case_name, slice_paths in case_slices.items():
            # Load mask and get header/affine
            mask_path = os.path.join(mask_folder, f"mask_body_contour_{case_name}.nii.gz")
            if not os.path.exists(mask_path):
                print(f"Mask not found for case {case_name}")
                continue
                
            mask_nifti = nib.load(mask_path)
            header = mask_nifti.header
            affine = mask_nifti.affine
            
            # Sort slices by slice number
            slice_paths.sort(key=lambda x: int(x.split('slice_')[-1].split('.')[0]))
            
            # Load and stack all slices
            ct_volume = []
            pred_ct_volume = []
            
            for slice_path in slice_paths:
                data = np.load(slice_path)
                ct_volume.append(data['CT'])
                pred_ct_volume.append(data['Pred_CT'])
            
            # Convert to numpy arrays and transform dimensions
            ct_volume = np.stack(ct_volume, axis=0)
            ct_volume = np.squeeze(ct_volume)
            ct_volume = np.moveaxis(ct_volume, 0, -1)
            
            pred_ct_volume = np.stack(pred_ct_volume, axis=0)
            pred_ct_volume = np.moveaxis(pred_ct_volume, 0, -1)
            
            # Compute metrics
            metrics = compute_metrics(ct_volume, pred_ct_volume)
            
            # Store results
            results.append({
                'Fold': fold,
                'Case': case_name,
                'MAE': metrics['MAE'],
                'SSIM': metrics['SSIM'],
                'PSNR': metrics['PSNR']
            })
            
            print(f"Processed case {case_name} in fold {fold}")
            
            # Save volumes as NIFTI
            output_dir = os.path.join(root_dir, f"{fold}_nifti")
            os.makedirs(output_dir, exist_ok=True)
            
            ct_nifti = nib.Nifti1Image(ct_volume, affine, header)
            nib.save(ct_nifti, os.path.join(output_dir, f"CT_{case_name}.nii.gz"))
            
            pred_ct_nifti = nib.Nifti1Image(pred_ct_volume, affine, header)
            nib.save(pred_ct_nifti, os.path.join(output_dir, f"Pred_CT_{case_name}.nii.gz"))
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    
    # Calculate mean metrics for each fold
    fold_means = df.groupby('Fold')[['MAE', 'SSIM', 'PSNR']].mean()
    
    # Calculate overall means
    overall_means = df[['MAE', 'SSIM', 'PSNR']].mean()
    
    # Create Excel writer
    excel_path = os.path.join(root_dir, 'metrics_results.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Write detailed results
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Write fold means
        fold_means.to_excel(writer, sheet_name='Fold Means')
        
        # Write overall means
        pd.DataFrame(overall_means).to_excel(writer, sheet_name='Overall Means')
    
    print(f"Results saved to {excel_path}")

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

