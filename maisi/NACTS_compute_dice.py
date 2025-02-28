import nibabel as nib
import numpy as np
import argparse
import os
import pandas as pd
from scipy.ndimage import distance_transform_edt
from ErasmusMC_NAC_TS_MAISI import T2M_mapping

# Default paths from ErasmusMC_NAC_TS_MAISI.py
NAC_TS_label_path = "combined_predictions/E4058_combined.nii.gz"
CT_TS_label_path = "NAC_CTAC_Spacing15/CTAC_E4058_TS.nii.gz"
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
        else:
            dice = jaccard = precision = recall = f1 = specificity = hausdorff = float('nan')
            if label in nac_unique_labels:
                status = "NAC only"
            else:
                status = "CT only"
        
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

def main():
    parser = argparse.ArgumentParser(description='Compute segmentation metrics between NAC and CT tissue segmentations')
    parser.add_argument('--nac_path', type=str, default=NAC_TS_label_path, help='Path to NAC tissue segmentation file')
    parser.add_argument('--ct_path', type=str, default=CT_TS_label_path, help='Path to CT tissue segmentation file')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save output files')
    parser.add_argument('--ts_label_path', type=str, default="TS_label.xlsx", help='Path to Excel file with TS label names')
    
    args = parser.parse_args()
    
    # Compute segmentation metrics and save to Excel
    excel_path = compute_segmentation_metrics(args.nac_path, args.ct_path, args.output_dir, args.ts_label_path)
    print(f"Segmentation metrics computation completed. Results saved to {excel_path}")

if __name__ == "__main__":
    main()
