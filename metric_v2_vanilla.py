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

def compute_metrics_with_mask(ct_volume, pred_ct_volume, mask_binary):
    """Compute metrics for a given mask alignment."""
    # Extract only the masked values
    ct_masked = ct_volume[mask_binary]
    pred_ct_masked = pred_ct_volume[mask_binary]
    
    # Compute MAE
    mae = np.mean(np.abs(ct_masked - pred_ct_masked)) * 4000
    
    # For SSIM and PSNR, compute slice by slice
    ssim_scores = []
    psnr_scores = []
    for z in range(ct_volume.shape[-1]):
        mask_slice = mask_binary[..., z]
        if np.any(mask_slice):  # Only compute if the slice has masked regions
            ct_slice = ct_volume[..., z]
            pred_slice = pred_ct_volume[..., z]
            
            # Apply mask to slices
            ct_slice_masked = ct_slice.copy()
            pred_slice_masked = pred_slice.copy()
            ct_slice_masked[~mask_slice] = 0
            pred_slice_masked[~mask_slice] = 0
            
            ssim_score = ssim(ct_slice_masked, pred_slice_masked, data_range=1.0)
            psnr_score = psnr(ct_slice_masked, pred_slice_masked, data_range=1.0)
            
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
    
    return {
        'MAE': mae,
        'SSIM': np.mean(ssim_scores) if ssim_scores else 0,
        'PSNR': np.mean(psnr_scores) if psnr_scores else 0
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
            mask_data = mask_nifti.get_fdata()
            mask_binary = mask_data > 0.5  # Create binary mask
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
            
            # Rescale volumes from [0.5, 1] to [0, 1]
            ct_volume = (ct_volume - 0.5) * 2
            pred_ct_volume = (pred_ct_volume - 0.5) * 2
            
            # Get dimensions
            ct_depth = ct_volume.shape[-1]
            mask_depth = mask_binary.shape[-1]
            diff = abs(ct_depth - mask_depth)
            
            # Create two versions of the mask
            if ct_depth < mask_depth:
                # Cut from start
                mask_start = mask_binary[..., diff:]
                metrics_start = compute_metrics_with_mask(ct_volume, pred_ct_volume, mask_start)
                
                # Cut from end
                mask_end = mask_binary[..., :ct_depth]
                metrics_end = compute_metrics_with_mask(ct_volume, pred_ct_volume, mask_end)
            else:
                print(f"Warning: CT depth ({ct_depth}) > mask depth ({mask_depth}) for case {case_name}")
                continue
            
            # Store results for both approaches
            results.append({
                'Fold': fold,
                'Case': case_name,
                'Alignment': 'Cut_Start',
                'MAE': metrics_start['MAE'],
                'SSIM': metrics_start['SSIM'],
                'PSNR': metrics_start['PSNR']
            })
            
            results.append({
                'Fold': fold,
                'Case': case_name,
                'Alignment': 'Cut_End',
                'MAE': metrics_end['MAE'],
                'SSIM': metrics_end['SSIM'],
                'PSNR': metrics_end['PSNR']
            })
            
            print(f"Processed case {case_name} in fold {fold}")
            
            # Save volumes as NIFTI for both alignments
            output_dir = os.path.join(root_dir, f"{fold}_nifti")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save for start-cut alignment
            ct_vis = ct_volume.copy()
            pred_ct_vis = pred_ct_volume.copy()
            ct_vis[~mask_start] = 0
            pred_ct_vis[~mask_start] = 0
            
            nib.save(nib.Nifti1Image(ct_vis, affine, header), 
                    os.path.join(output_dir, f"CT_{case_name}_cut_start.nii.gz"))
            nib.save(nib.Nifti1Image(pred_ct_vis, affine, header), 
                    os.path.join(output_dir, f"Pred_CT_{case_name}_cut_start.nii.gz"))
            
            # Save for end-cut alignment
            ct_vis = ct_volume.copy()
            pred_ct_vis = pred_ct_volume.copy()
            ct_vis[~mask_end] = 0
            pred_ct_vis[~mask_end] = 0
            
            nib.save(nib.Nifti1Image(ct_vis, affine, header), 
                    os.path.join(output_dir, f"CT_{case_name}_cut_end.nii.gz"))
            nib.save(nib.Nifti1Image(pred_ct_vis, affine, header), 
                    os.path.join(output_dir, f"Pred_CT_{case_name}_cut_end.nii.gz"))
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    
    # Calculate mean metrics for each fold and alignment
    fold_means = df.groupby(['Fold', 'Alignment'])[['MAE', 'SSIM', 'PSNR']].mean()
    
    # Calculate overall means for each alignment
    overall_means = df.groupby('Alignment')[['MAE', 'SSIM', 'PSNR']].mean()
    
    # Create Excel writer
    excel_path = os.path.join(root_dir, 'metrics_results.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Write detailed results
        df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Write fold means
        fold_means.to_excel(writer, sheet_name='Fold Means')
        
        # Write overall means
        overall_means.to_excel(writer, sheet_name='Overall Means')
    
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

