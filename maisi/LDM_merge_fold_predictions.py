import os
import argparse
import json
import time
import nibabel as nib
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# Constants for normalization (same as in LDM_inference.py)
HU_MIN = -1024
HU_MAX = 1976

def setup_logger(log_dir):
    """Set up a logger that writes to a text file"""
    import datetime
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"merge_fold_log_{timestamp}.txt")
    
    def log_message(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_entry)
        
        # Write to log file
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
    
    return log_message

def calculate_metrics(ct_data, pred_data, masks, region_names, logger=None):
    """Calculate MAE for different regions using masks"""
    metrics = {}
    
    for i, region in enumerate(region_names):
        gt_mask = masks[i]
        
        # Calculate MAE
        mae = np.mean(np.abs(ct_data[gt_mask] - pred_data[gt_mask]))
        
        metrics[region] = {
            'mae': mae
        }
        
        if logger:
            logger(f"Region {region}: MAE={mae:.4f} HU")
    
    return metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Merge predictions from multiple folds and calculate metrics')
    parser.add_argument('--data_dir', type=str, default='LDM_adapter', help='Root directory containing the data')
    parser.add_argument('--fold_predictions', type=str, nargs='+', default=[
        'LDM_adapter/results/predictions/fold_1',
        'LDM_adapter/results/predictions/fold_2',
        'LDM_adapter/results/predictions/fold_3',
        'LDM_adapter/results/predictions/fold_4'
    ], help='Directories containing fold predictions')
    parser.add_argument('--mask_dir', type=str, default='LDM_adapter/results/masks/fold_1', 
                        help='Directory containing masks (reuse from one fold)')
    parser.add_argument('--output_dir', type=str, default='LDM_adapter/results/predictions/ensemble', 
                        help='Directory to save merged predictions')
    parser.add_argument('--metrics_dir', type=str, default='LDM_adapter/results/metrics/ensemble', 
                        help='Directory to save metrics')
    parser.add_argument('--log_dir', type=str, default='LDM_adapter/results/logs/ensemble', 
                        help='Directory to save logs')
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    logger(f"Starting ensemble prediction merging from {len(args.fold_predictions)} folds")
    
    # Identify common test cases across all folds
    all_cases = set()
    for fold_dir in args.fold_predictions:
        # Find all axial predictions in this fold
        axial_files = glob.glob(os.path.join(fold_dir, "*_axial.nii.gz"))
        case_ids = [os.path.basename(f).split("_axial.nii.gz")[0] for f in axial_files]
        if not all_cases:
            all_cases = set(case_ids)
        else:
            all_cases = all_cases.intersection(set(case_ids))
    
    case_ids_list = sorted(list(all_cases))
    logger(f"Found {len(case_ids_list)} common test cases across all folds")
    
    # Prepare for storing metrics
    region_names = ["body", "soft", "bone"]
    all_metrics_mean = {region: {'mae': []} for region in region_names}
    all_metrics_median = {region: {'mae': []} for region in region_names}
    
    case_metrics_mean = {}
    case_metrics_median = {}
    
    # Process each test case
    for case_id in tqdm(case_ids_list, desc="Processing cases"):
        logger(f"Processing case: {case_id}")
        
        # Load all predictions for this case from all folds
        all_predictions = []
        
        for fold_dir in args.fold_predictions:
            # Load axial, coronal, and sagittal predictions from this fold
            axial_path = os.path.join(fold_dir, f"{case_id}_axial.nii.gz")
            coronal_path = os.path.join(fold_dir, f"{case_id}_coronal.nii.gz")
            sagittal_path = os.path.join(fold_dir, f"{case_id}_sagittal.nii.gz")
            merged_path = os.path.join(fold_dir, f"{case_id}_merged.nii.gz")
            
            try:
                axial_nifti = nib.load(axial_path)
                coronal_nifti = nib.load(coronal_path)
                sagittal_nifti = nib.load(sagittal_path)
                merged_nifti = nib.load(merged_path)
                
                # Store predictions and metadata from the first fold
                if not all_predictions:
                    affine = axial_nifti.affine
                    header = axial_nifti.header
                
                # Add predictions to list
                all_predictions.extend([
                    axial_nifti.get_fdata(),
                    coronal_nifti.get_fdata(),
                    sagittal_nifti.get_fdata(),
                    merged_nifti.get_fdata()
                ])
                
                logger(f"Loaded predictions from {fold_dir}")
            except Exception as e:
                logger(f"Error loading predictions from {fold_dir}: {str(e)}")
                continue
        
        if not all_predictions:
            logger(f"No predictions found for case {case_id}, skipping")
            continue
        
        # Stack all predictions
        stacked_preds = np.stack(all_predictions, axis=0)
        
        # Compute mean and median super-ensembles
        mean_pred = np.mean(stacked_preds, axis=0)
        median_pred = np.median(stacked_preds, axis=0)
        
        # Save mean prediction
        mean_output_path = os.path.join(args.output_dir, f"{case_id}_mean.nii.gz")
        mean_nib_img = nib.Nifti1Image(mean_pred, affine, header)
        nib.save(mean_nib_img, mean_output_path)
        logger(f"Mean ensemble prediction saved to {mean_output_path}")
        
        # Save median prediction
        median_output_path = os.path.join(args.output_dir, f"{case_id}_median.nii.gz")
        median_nib_img = nib.Nifti1Image(median_pred, affine, header)
        nib.save(median_nib_img, median_output_path)
        logger(f"Median ensemble prediction saved to {median_output_path}")
        
        # Load reference CT for metrics calculation
        ct_path = f"{args.data_dir}/data/CT/CTAC_{case_id}_cropped.nii.gz"
        try:
            ct_nifti = nib.load(ct_path)
            reference_volume = ct_nifti.get_fdata()
            logger(f"Loaded reference CT: {reference_volume.shape}")
        except Exception as e:
            logger(f"Error loading reference CT {ct_path}: {str(e)}")
            continue
        
        # Load masks
        try:
            body_mask_path = os.path.join(args.mask_dir, f"{case_id}_body.nii.gz")
            soft_mask_path = os.path.join(args.mask_dir, f"{case_id}_soft.nii.gz")
            bone_mask_path = os.path.join(args.mask_dir, f"{case_id}_bone.nii.gz")
            
            body_mask = nib.load(body_mask_path).get_fdata()
            soft_mask = nib.load(soft_mask_path).get_fdata()
            bone_mask = nib.load(bone_mask_path).get_fdata()
            
            masks = [body_mask, soft_mask, bone_mask]
            logger(f"Loaded masks for case {case_id}")
        except Exception as e:
            logger(f"Error loading masks for case {case_id}: {str(e)}")
            continue
        
        # Calculate metrics for mean prediction
        mean_metrics = calculate_metrics(reference_volume, mean_pred, masks, region_names, logger)
        case_metrics_mean[case_id] = mean_metrics
        
        # Add metrics to overall statistics
        for region in region_names:
            all_metrics_mean[region]['mae'].append(mean_metrics[region]['mae'])
        
        # Calculate metrics for median prediction
        median_metrics = calculate_metrics(reference_volume, median_pred, masks, region_names, logger)
        case_metrics_median[case_id] = median_metrics
        
        # Add metrics to overall statistics
        for region in region_names:
            all_metrics_median[region]['mae'].append(median_metrics[region]['mae'])
    
    # Calculate average metrics across all cases
    avg_metrics_mean = {region: {
        'mae': np.mean(all_metrics_mean[region]['mae'])
    } for region in region_names}
    
    avg_metrics_median = {region: {
        'mae': np.mean(all_metrics_median[region]['mae'])
    } for region in region_names}
    
    # Print average metrics
    logger("Average metrics across all cases (Mean Ensemble):")
    for region in region_names:
        logger(f"Region {region}: MAE={avg_metrics_mean[region]['mae']:.4f} HU")
    
    logger("Average metrics across all cases (Median Ensemble):")
    for region in region_names:
        logger(f"Region {region}: MAE={avg_metrics_median[region]['mae']:.4f} HU")
    
    # Save metrics to JSON file
    metrics_output = {
        "mean_ensemble": {
            "metrics_by_case": case_metrics_mean,
            "average_metrics": avg_metrics_mean
        },
        "median_ensemble": {
            "metrics_by_case": case_metrics_median,
            "average_metrics": avg_metrics_median
        }
    }
    
    metrics_path = os.path.join(args.metrics_dir, "ensemble_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger(f"Metrics saved to {metrics_path}")
    
    # Save metrics to Excel file
    excel_path = os.path.join(args.metrics_dir, "ensemble_metrics.xlsx")
    
    # Create DataFrames for case metrics (mean ensemble)
    mean_case_metrics_data = []
    for case_id, metrics in case_metrics_mean.items():
        row = {"case_id": case_id}
        for region, region_metrics in metrics.items():
            for metric_name, value in region_metrics.items():
                row[f"{region}_{metric_name}"] = value
        mean_case_metrics_data.append(row)
    
    mean_case_df = pd.DataFrame(mean_case_metrics_data)
    
    # Create DataFrames for case metrics (median ensemble)
    median_case_metrics_data = []
    for case_id, metrics in case_metrics_median.items():
        row = {"case_id": case_id}
        for region, region_metrics in metrics.items():
            for metric_name, value in region_metrics.items():
                row[f"{region}_{metric_name}"] = value
        median_case_metrics_data.append(row)
    
    median_case_df = pd.DataFrame(median_case_metrics_data)
    
    # Create a DataFrame for average metrics (mean ensemble)
    mean_avg_data = []
    for region, region_metrics in avg_metrics_mean.items():
        for metric_name, value in region_metrics.items():
            mean_avg_data.append({"region": region, "metric": metric_name, "value": value})
    
    mean_avg_df = pd.DataFrame(mean_avg_data)
    
    # Create a DataFrame for average metrics (median ensemble)
    median_avg_data = []
    for region, region_metrics in avg_metrics_median.items():
        for metric_name, value in region_metrics.items():
            median_avg_data.append({"region": region, "metric": metric_name, "value": value})
    
    median_avg_df = pd.DataFrame(median_avg_data)
    
    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        mean_case_df.to_excel(writer, sheet_name='Mean Case Metrics', index=False)
        median_case_df.to_excel(writer, sheet_name='Median Case Metrics', index=False)
        mean_avg_df.to_excel(writer, sheet_name='Mean Average Metrics', index=False)
        median_avg_df.to_excel(writer, sheet_name='Median Average Metrics', index=False)
    
    logger(f"Metrics saved to Excel: {excel_path}")
    logger("Ensemble merging completed successfully!")

if __name__ == "__main__":
    main() 