import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

# Define input/output directories
CTAC_maisi_src_folder = "NAC_CTAC_Spacing15/inference_20250128_noon"
CTAC_maisi_dst_folder = "James_36/CTAC_maisi"
CT_bed_folder = "James_36/CT_bed/"

case_name_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
]

# HU threshold for body contour
body_contour_HU_th = -500

print(f"Starting processing {len(case_name_list)} cases")

for idx, case_name in enumerate(case_name_list, 1):
    print(f"[{idx}/{len(case_name_list)}] Processing case: {case_name}")

    # Load input files
    CTAC_maisi_src_path = f"{CTAC_maisi_src_folder}/CTAC_maisi_{case_name}_v1_3dresample.nii.gz"
    CT_bed_path = sorted(glob.glob(f"{CT_bed_folder}*_{case_name[1:]}_*.nii"))[0]
    
    print(f"  [{idx}/{len(case_name_list)}] Loading nifti files...")
    CT_bed_nifti = nib.load(CT_bed_path)
    CTAC_maisi_nifti = nib.load(CTAC_maisi_src_path)
    
    # Get data arrays
    CT_bed_data = CT_bed_nifti.get_fdata()
    CTAC_maisi_data = CTAC_maisi_nifti.get_fdata()

    print(f"  Data shapes - CT_bed: {CT_bed_data.shape}, CTAC_maisi: {CTAC_maisi_data.shape}")

    # Pad CTAC_maisi to match CT_bed dimensions
    pad_width = ((CT_bed_data.shape[0] - CTAC_maisi_data.shape[0])//2,
                 (CT_bed_data.shape[1] - CTAC_maisi_data.shape[1])//2)
    
    CTAC_maisi_data = np.pad(CTAC_maisi_data, 
                            ((pad_width[0], CT_bed_data.shape[0]-CTAC_maisi_data.shape[0]-pad_width[0]),
                             (pad_width[1], CT_bed_data.shape[1]-CTAC_maisi_data.shape[1]-pad_width[1]),
                             (0, 0)), 
                            mode='constant', constant_values=-1024)

    print(f"  After padding - CT_bed: {CT_bed_data.shape}, CTAC_maisi: {CTAC_maisi_data.shape}")

    # Create body contour array
    CTAC_maisi_body_contour = np.zeros_like(CTAC_maisi_data)
    
    print(f"  [{idx}/{len(case_name_list)}] Processing {CT_bed_data.shape[2]} slices for body contours...")
    # Process each z-slice
    for z in range(CT_bed_data.shape[2]):
        # Create mask from CTAC data using HU threshold
        CTAC_maisi_mask = CTAC_maisi_data[:,:,z] > body_contour_HU_th
        
        # Fill holes in the mask
        CTAC_maisi_filled_mask = binary_fill_holes(CTAC_maisi_mask)
        
        # Save the filled mask
        CTAC_maisi_body_contour[:,:,z] = CTAC_maisi_filled_mask

    print(f"  Saving body contour mask...")
    # Save body contour mask
    CTAC_maisi_contour_nifti = nib.Nifti1Image(CTAC_maisi_body_contour, CTAC_maisi_nifti.affine, CTAC_maisi_nifti.header)
    CTAC_maisi_contour_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v2_body_contour.nii.gz"
    nib.save(CTAC_maisi_contour_nifti, CTAC_maisi_contour_path)

    print(f"  Creating masked bed data...")
    # Create masked version of CT bed data
    CTAC_maisi_v3_bed = CT_bed_data.copy()

    # Replace CT bed data with CTAC data where body contour is True
    CTAC_maisi_v3_bed[CTAC_maisi_body_contour == 1] = CTAC_maisi_data[CTAC_maisi_body_contour == 1]

    print(f"  Saving masked bed data...")
    # Save masked bed data
    CTAC_maisi_v3_bed_nifti = nib.Nifti1Image(CTAC_maisi_v3_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    CTAC_maisi_v3_bed_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v3_bed.nii.gz"
    nib.save(CTAC_maisi_v3_bed_nifti, CTAC_maisi_v3_bed_path)
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")