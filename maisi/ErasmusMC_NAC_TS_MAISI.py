NAC_TS_label_path = "combined_predictions/E4058_combined.nii.gz" # no 200
NAC_TS_body_contour_PET_path = "NAC_body_contour_thresholds/NAC_E4058_final_contour.nii.gz"
NAC_TS_body_contour_CT_path = "James_36/CT_mask/mask_body_contour_E4058.nii.gz"
CT_TS_label_path = "NAC_CTAC_Spacing15/CTAC_E4058_TS.nii.gz"

# Add imports for computing Dice coefficients
import nibabel as nib
import numpy as np
from scipy.spatial.distance import dice

def compute_dice_coefficient(y_true, y_pred):
    """
    Compute Dice coefficient: 2*|X∩Y|/(|X|+|Y|)
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# Load the segmentation files
print("Loading segmentation files...")
nac_ts_img = nib.load(NAC_TS_label_path)
ct_ts_img = nib.load(CT_TS_label_path)

nac_ts_data = nac_ts_img.get_fdata()
ct_ts_data = ct_ts_img.get_fdata()

# Find unique class labels in both segmentations
nac_unique_labels = np.unique(nac_ts_data).astype(int)
ct_unique_labels = np.unique(ct_ts_data).astype(int)

print(f"Unique labels in NAC TS: {nac_unique_labels}")
print(f"Unique labels in CT TS: {ct_unique_labels}")

# Find common labels to compute Dice for
common_labels = np.intersect1d(nac_unique_labels, ct_unique_labels)
all_labels = np.union1d(nac_unique_labels, ct_unique_labels)

print(f"Common labels: {common_labels}")
print(f"All labels: {all_labels}")

# Compute Dice coefficient for each class
print("\nDice coefficients for each class:")
for label in all_labels:
    if label == 0:  # Skip background
        continue
        
    # Create binary masks for this class
    nac_mask = (nac_ts_data == label).astype(int)
    ct_mask = (ct_ts_data == label).astype(int)
    
    # Compute Dice
    if label in common_labels:
        dice_score = compute_dice_coefficient(nac_mask, ct_mask)
        print(f"Class {label}: {dice_score:.4f}")
    else:
        if label in nac_unique_labels:
            print(f"Class {label}: Only in NAC segmentation")
        else:
            print(f"Class {label}: Only in CT segmentation")

# Compute overall Dice (considering all non-zero labels as foreground)
nac_foreground = (nac_ts_data > 0).astype(int)
ct_foreground = (ct_ts_data > 0).astype(int)
overall_dice = compute_dice_coefficient(nac_foreground, ct_foreground)
print(f"\nOverall Dice (all foreground vs background): {overall_dice:.4f}")

