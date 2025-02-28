import nibabel as nib
import numpy as np
import argparse
import os
import pandas as pd
from scipy.ndimage import distance_transform_edt
from ErasmusMC_NAC_TS_MAISI import T2M_mapping

case_name_list = [
    # 'E4058',
    'E4055',          'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
]

# Default paths and directories
DEFAULT_NAC_TS_DIR = "combined_predictions"
DEFAULT_CT_TS_DIR = "NAC_CTAC_Spacing15"
DEFAULT_OUTPUT_DIR = "ErasmusMC"

def compute_dice_coefficient(y_true, y_pred):
    """
    Compute Dice coefficient: 2*|X∩Y|/(|X|+|Y|)
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def compute_jaccard_index(y_true, y_pred):
    """
    Compute Jaccard Index (IoU): |X∩Y|/|X∪Y|
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 0.0
    return intersection / union

def compute_precision(y_true, y_pred):
    """
    Compute Precision: TP/(TP+FP)
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives

def compute_recall(y_true, y_pred):
    """
    Compute Recall (Sensitivity): TP/(TP+FN)
    """
    true_positives = np.sum(y_true * y_pred)
    actual_positives = np.sum(y_true)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives

def compute_f1_score(precision, recall):
    """
    Compute F1 Score: 2*(Precision*Recall)/(Precision+Recall)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_specificity(y_true, y_pred, total_pixels):
    """
    Compute Specificity: TN/(TN+FP)
    """
    true_negatives = total_pixels - np.sum(y_true | y_pred)
    false_positives = np.sum(y_pred) - np.sum(y_true * y_pred)
    if true_negatives + false_positives == 0:
        return 0.0
    return true_negatives / (true_negatives + false_positives)

def compute_hausdorff_distance(y_true, y_pred, voxel_spacing=None):
    """
    Compute Hausdorff Distance between two binary masks
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        voxel_spacing: Voxel spacing in mm (optional)
        
    Returns:
        float: Hausdorff distance
    """
    # If either mask is empty, return infinity
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return float('inf')
    
    # Get the boundary of each mask
    y_true_boundary = get_boundary(y_true)
    y_pred_boundary = get_boundary(y_pred)
    
    # If either boundary is empty, return infinity
    if np.sum(y_true_boundary) == 0 or np.sum(y_pred_boundary) == 0:
        return float('inf')
    
    # Compute distance transforms
    dt_true = distance_transform_edt(~y_true_boundary, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~y_pred_boundary, sampling=voxel_spacing)
    
    # Compute Hausdorff distance
    hausdorff_true_to_pred = np.max(dt_true * y_pred_boundary)
    hausdorff_pred_to_true = np.max(dt_pred * y_true_boundary)
    
    return max(hausdorff_true_to_pred, hausdorff_pred_to_true)

def get_boundary(binary_mask):
    """
    Extract the boundary of a binary mask
    """
    from scipy.ndimage import binary_erosion
    
    # Erode the mask to get the inner region
    eroded = binary_erosion(binary_mask)
    
    # The boundary is the difference between the original mask and the eroded mask
    boundary = binary_mask & ~eroded
    
    return boundary

def compute_segmentation_metrics(nac_path, ct_path, output_dir=DEFAULT_OUTPUT_DIR, ts_label_path="TS_label.xlsx"):
    """
    Compute various segmentation metrics between NAC and CT tissue segmentations
    and save results to an Excel file
    
    Args:
        nac_path: Path to the NAC tissue segmentation file
        ct_path: Path to the CT tissue segmentation file
        output_dir: Directory to save results
        ts_label_path: Path to Excel file containing TS label names
        
    Returns:
        str: Path to the saved Excel file
    """
    # Check if files exist
    if not os.path.exists(nac_path):
        raise FileNotFoundError(f"NAC segmentation file not found: {nac_path}")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT segmentation file not found: {ct_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TS label names if available
    ts_label_names = {}
    try:
        if os.path.exists(ts_label_path):
            print(f"Loading TS label names from {ts_label_path}...")
            label_df = pd.read_excel(ts_label_path)
            
            # Check for the expected columns - support both formats
            if 'TS_Index' in label_df.columns and 'TS_label' in label_df.columns:
                # Format: TS_Index, TS_label
                for _, row in label_df.iterrows():
                    ts_label_names[int(row['TS_Index'])] = row['TS_label']
                print(f"Loaded {len(ts_label_names)} label names from TS_Index/TS_label columns")
            elif 'Label' in label_df.columns and 'Name' in label_df.columns:
                # Alternative format: Label, Name
                for _, row in label_df.iterrows():
                    ts_label_names[int(row['Label'])] = row['Name']
                print(f"Loaded {len(ts_label_names)} label names from Label/Name columns")
            else:
                print(f"Warning: Expected columns not found in {ts_label_path}")
                print(f"Available columns: {label_df.columns.tolist()}")
        else:
            print(f"Warning: TS label file {ts_label_path} not found. Will use numeric labels only.")
    except Exception as e:
        print(f"Error loading TS label names: {e}")
        print("Will use numeric labels only.")
    
    # Load the segmentation files
    print("Loading segmentation files...")
    nac_ts_img = nib.load(nac_path)
    ct_ts_img = nib.load(ct_path)

    nac_ts_data = nac_ts_img.get_fdata()
    ct_ts_data = ct_ts_img.get_fdata()
    
    # Get voxel spacing for Hausdorff distance calculation
    voxel_spacing = nac_ts_img.header.get_zooms()
    
    # Find unique class labels in both segmentations
    nac_unique_labels = np.unique(nac_ts_data).astype(int)
    ct_unique_labels = np.unique(ct_ts_data).astype(int)

    print(f"Unique labels in NAC TS: {nac_unique_labels}")
    print(f"Unique labels in CT TS: {ct_unique_labels}")

    # Find common labels to compute metrics for
    common_labels = np.intersect1d(nac_unique_labels, ct_unique_labels)
    all_labels = np.union1d(nac_unique_labels, ct_unique_labels)

    print(f"Common labels: {common_labels}")
    print(f"All labels: {all_labels}")
    
    # Total number of pixels/voxels in the image
    total_pixels = nac_ts_data.size
    
    # Prepare data for Excel
    results_data = []
    
    # Compute metrics for each class
    print("\nComputing segmentation metrics for each class...")
    print("=" * 80)
    print(f"{'TS Label':<8} {'Class Name':<20} {'Status':<10} {'Dice':<10} {'Jaccard':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Specificity':<10} {'Hausdorff':<10}")
    print("-" * 80)
    
    for label in all_labels:
        if label == 0:  # Skip background
            continue
            
        # Create binary masks for this class
        nac_mask = (nac_ts_data == label).astype(int)
        ct_mask = (ct_ts_data == label).astype(int)
        
        # Get MAISI label if available
        maisi_label = T2M_mapping.get(label, "N/A") if label in T2M_mapping else "N/A"
        
        # Get class name if available
        class_name = ts_label_names.get(label, "Unknown")
        
        # Compute metrics if label exists in both segmentations
        if label in common_labels:
            dice = compute_dice_coefficient(nac_mask, ct_mask)
            jaccard = compute_jaccard_index(nac_mask, ct_mask)
            precision = compute_precision(ct_mask, nac_mask)  # CT as ground truth
            recall = compute_recall(ct_mask, nac_mask)  # CT as ground truth
            f1 = compute_f1_score(precision, recall)
            specificity = compute_specificity(ct_mask, nac_mask, total_pixels)
            
            try:
                hausdorff = compute_hausdorff_distance(ct_mask, nac_mask, voxel_spacing)
            except:
                hausdorff = float('nan')
                
            status = "Common"
            
            # Print metrics with 4 decimal places
            print(f"{label:<8} {class_name:<20} {status:<10} {dice:.4f}     {jaccard:.4f}     {precision:.4f}     {recall:.4f}     {f1:.4f}     {specificity:.4f}     {hausdorff:.4f}")
        else:
            dice = jaccard = precision = recall = f1 = specificity = hausdorff = float('nan')
            if label in nac_unique_labels:
                status = "NAC only"
            else:
                status = "CT only"
            
            # Print status for labels that exist in only one segmentation
            print(f"{label:<8} {class_name:<20} {status:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        # Add results to data list
        results_data.append({
            'TS_Label': int(label),
            'Class_Name': class_name,
            'MAISI_Label': maisi_label,
            'Status': status,
            'Dice': dice,
            'Jaccard': jaccard,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Specificity': specificity,
            'Hausdorff_Distance': hausdorff
        })
    
    # Compute overall metrics (considering all non-zero labels as foreground)
    nac_foreground = (nac_ts_data > 0).astype(int)
    ct_foreground = (ct_ts_data > 0).astype(int)
    
    overall_dice = compute_dice_coefficient(nac_foreground, ct_foreground)
    overall_jaccard = compute_jaccard_index(nac_foreground, ct_foreground)
    overall_precision = compute_precision(ct_foreground, nac_foreground)
    overall_recall = compute_recall(ct_foreground, nac_foreground)
    overall_f1 = compute_f1_score(overall_precision, overall_recall)
    overall_specificity = compute_specificity(ct_foreground, nac_foreground, total_pixels)
    
    try:
        overall_hausdorff = compute_hausdorff_distance(ct_foreground, nac_foreground, voxel_spacing)
    except:
        overall_hausdorff = float('nan')
    
    # Print overall metrics
    print("-" * 80)
    print(f"{'Overall':<8} {'All Tissues':<20} {'Foreground':<10} {overall_dice:.4f}     {overall_jaccard:.4f}     {overall_precision:.4f}     {overall_recall:.4f}     {overall_f1:.4f}     {overall_specificity:.4f}     {overall_hausdorff:.4f}")
    print("=" * 80)
    
    # Add overall results to data list
    results_data.append({
        'TS_Label': 'Overall',
        'Class_Name': 'All Tissues',
        'MAISI_Label': 'N/A',
        'Status': 'Foreground',
        'Dice': overall_dice,
        'Jaccard': overall_jaccard,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1_Score': overall_f1,
        'Specificity': overall_specificity,
        'Hausdorff_Distance': overall_hausdorff
    })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    
    # Get base filename without extension for the output file
    base_filename = os.path.splitext(os.path.basename(nac_path))[0]
    if base_filename.endswith('.nii'):  # Handle .nii.gz case
        base_filename = os.path.splitext(base_filename)[0]
    
    excel_path = os.path.join(output_dir, f"{base_filename}_segmentation_metrics.xlsx")
    
    # Save to Excel
    df.to_excel(excel_path, index=False)
    print(f"Segmentation metrics saved to {excel_path}")
    
    # Also save a summary text file
    summary_path = os.path.join(output_dir, f"{base_filename}_metrics_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Segmentation Metrics Summary\n")
        f.write("===========================\n\n")
        f.write(f"NAC file: {nac_path}\n")
        f.write(f"CT file: {ct_path}\n\n")
        f.write("Overall Metrics (Foreground vs Background):\n")
        f.write(f"Dice: {overall_dice:.4f}\n")
        f.write(f"Jaccard (IoU): {overall_jaccard:.4f}\n")
        f.write(f"Precision: {overall_precision:.4f}\n")
        f.write(f"Recall: {overall_recall:.4f}\n")
        f.write(f"F1 Score: {overall_f1:.4f}\n")
        f.write(f"Specificity: {overall_specificity:.4f}\n")
        f.write(f"Hausdorff Distance: {overall_hausdorff:.4f}\n")
    
    print(f"Summary saved to {summary_path}")
    return excel_path

def process_all_cases(case_list, nac_ts_dir=DEFAULT_NAC_TS_DIR, ct_ts_dir=DEFAULT_CT_TS_DIR, 
                      output_dir=DEFAULT_OUTPUT_DIR, ts_label_path="TS_label.xlsx"):
    """
    Process all cases in the given list, computing segmentation metrics for each
    
    Args:
        case_list: List of case names to process
        nac_ts_dir: Directory containing NAC tissue segmentation files
        ct_ts_dir: Directory containing CT tissue segmentation files
        output_dir: Directory to save results
        ts_label_path: Path to Excel file containing TS label names
        
    Returns:
        list: Paths to all saved Excel files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store paths to all saved Excel files
    excel_paths = []
    
    # Process each case
    for case_name in case_list:
        print(f"\n{'='*80}")
        print(f"Processing case: {case_name}")
        print(f"{'='*80}")
        
        # Construct paths for this case
        nac_path = os.path.join(nac_ts_dir, f"{case_name}_combined.nii.gz")
        ct_path = os.path.join(ct_ts_dir, f"CTAC_{case_name}_TS.nii.gz")
        
        # Check if files exist
        if not os.path.exists(nac_path):
            print(f"Warning: NAC segmentation file not found: {nac_path}")
            print(f"Skipping case {case_name}")
            continue
            
        if not os.path.exists(ct_path):
            print(f"Warning: CT segmentation file not found: {ct_path}")
            print(f"Skipping case {case_name}")
            continue
        
        # Create case-specific output directory
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        try:
            # Compute segmentation metrics for this case
            excel_path = compute_segmentation_metrics(nac_path, ct_path, case_output_dir, ts_label_path)
            excel_paths.append(excel_path)
            print(f"Successfully processed case {case_name}")
        except Exception as e:
            print(f"Error processing case {case_name}: {e}")
    
    # Create a summary Excel file with results from all cases
    if excel_paths:
        try:
            create_summary_report(excel_paths, output_dir)
        except Exception as e:
            print(f"Error creating summary report: {e}")
    
    return excel_paths

def create_summary_report(excel_paths, output_dir):
    """
    Create a summary report combining results from all cases
    
    Args:
        excel_paths: List of paths to individual Excel files
        output_dir: Directory to save the summary report
    """
    print("\nCreating summary report...")
    
    # List to store DataFrames from all Excel files
    all_dfs = []
    
    # Process each Excel file
    for excel_path in excel_paths:
        try:
            # Extract case name from the path
            path_parts = excel_path.split(os.sep)
            case_name = path_parts[-2] if len(path_parts) >= 2 else "Unknown"
            
            # Read the Excel file
            df = pd.read_excel(excel_path)
            
            # Add case name column
            df['Case'] = case_name
            
            # Add to list
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {excel_path}: {e}")
    
    if not all_dfs:
        print("No data to create summary report")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined DataFrame to Excel
    summary_path = os.path.join(output_dir, "all_cases_segmentation_metrics.xlsx")
    combined_df.to_excel(summary_path, index=False)
    print(f"Summary report saved to {summary_path}")
    
    # Create pivot tables for key metrics
    try:
        # Filter for overall metrics
        overall_df = combined_df[combined_df['TS_Label'] == 'Overall']
        
        # Create pivot table with cases as rows and metrics as columns
        pivot_df = pd.pivot_table(
            overall_df, 
            values=['Dice', 'Jaccard', 'Precision', 'Recall', 'F1_Score'],
            index=['Case'],
            aggfunc='mean'
        )
        
        # Add summary statistics
        pivot_df.loc['Mean'] = pivot_df.mean()
        pivot_df.loc['Median'] = pivot_df.median()
        pivot_df.loc['Std Dev'] = pivot_df.std()
        
        # Save pivot table to Excel
        pivot_path = os.path.join(output_dir, "overall_metrics_summary.xlsx")
        pivot_df.to_excel(pivot_path)
        print(f"Overall metrics summary saved to {pivot_path}")
        
    except Exception as e:
        print(f"Error creating pivot tables: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compute segmentation metrics between NAC and CT tissue segmentations')
    parser.add_argument('--nac_path', type=str, help='Path to NAC tissue segmentation file (for single case)')
    parser.add_argument('--ct_path', type=str, help='Path to CT tissue segmentation file (for single case)')
    parser.add_argument('--nac_dir', type=str, default=DEFAULT_NAC_TS_DIR, help='Directory containing NAC tissue segmentation files')
    parser.add_argument('--ct_dir', type=str, default=DEFAULT_CT_TS_DIR, help='Directory containing CT tissue segmentation files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save output files')
    parser.add_argument('--ts_label_path', type=str, default="TS_label.xlsx", help='Path to Excel file with TS label names')
    parser.add_argument('--case', type=str, help='Process a specific case name')
    parser.add_argument('--all_cases', action='store_true', help='Process all cases in the case_name_list')
    
    args = parser.parse_args()
    
    # Check if processing a single case with explicit paths
    if args.nac_path and args.ct_path:
        print("Processing single case with provided paths...")
        excel_path = compute_segmentation_metrics(args.nac_path, args.ct_path, args.output_dir, args.ts_label_path)
        print(f"Segmentation metrics computation completed. Results saved to {excel_path}")
    
    # Check if processing a specific case from the list
    elif args.case:
        print(f"Processing case: {args.case}")
        case_list = [args.case]
        process_all_cases(case_list, args.nac_dir, args.ct_dir, args.output_dir, args.ts_label_path)
    
    # Check if processing all cases
    elif args.all_cases:
        print(f"Processing all {len(case_name_list)} cases...")
        process_all_cases(case_name_list, args.nac_dir, args.ct_dir, args.output_dir, args.ts_label_path)
    
    # Default behavior: process all cases
    else:
        print(f"No specific options provided. Processing all {len(case_name_list)} cases...")
        process_all_cases(case_name_list, args.nac_dir, args.ct_dir, args.output_dir, args.ts_label_path)

if __name__ == "__main__":
    main()
