import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

# List of cases to process
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

# Define input/output directories
sCT1_src_folder = "James_36/sCT1_MLAA"
sCT2_src_folder = "James_36/sCT2_MLAA"
CT_bed_folder = "CTAC_bed/"

# HU threshold for body contour
body_contour_HU_th = -500

print(f"Starting processing {len(case_name_list)} cases")

for idx, case_name in enumerate(case_name_list, 1):
    print(f"[{idx}/{len(case_name_list)}] Processing case: {case_name}")

    # Load input files
    sCT1_src_path = f"{sCT1_src_folder}/sCT1_{case_name}_v1_3dresample.nii.gz"
    sCT2_src_path = f"{sCT2_src_folder}/sCT2_{case_name}_v1_3dresample.nii.gz"
    CT_bed_path = sorted(glob.glob(f"{CT_bed_folder}*_{case_name[1:]}_*.nii"))[0]
    
    print(f"  [{idx}/{len(case_name_list)}] Loading nifti files...")
    CT_bed_nifti = nib.load(CT_bed_path)
    sCT1_nifti = nib.load(sCT1_src_path)
    sCT2_nifti = nib.load(sCT2_src_path)
    
    # Get data arrays
    CT_bed_data = CT_bed_nifti.get_fdata()
    sCT1_data = sCT1_nifti.get_fdata()
    sCT2_data = sCT2_nifti.get_fdata()

    print(f"  Data shapes - CT_bed: {CT_bed_data.shape}, sCT1: {sCT1_data.shape}, sCT2: {sCT2_data.shape}")

    # Pad sCT1 and sCT2 to match CT_bed dimensions
    pad_width1 = ((CT_bed_data.shape[0] - sCT1_data.shape[0])//2,
                  (CT_bed_data.shape[1] - sCT1_data.shape[1])//2)
    pad_width2 = ((CT_bed_data.shape[0] - sCT2_data.shape[0])//2,
                  (CT_bed_data.shape[1] - sCT2_data.shape[1])//2)
    
    sCT1_data = np.pad(sCT1_data, 
                       ((pad_width1[0], CT_bed_data.shape[0]-sCT1_data.shape[0]-pad_width1[0]),
                        (pad_width1[1], CT_bed_data.shape[1]-sCT1_data.shape[1]-pad_width1[1]),
                        (0, 0)), 
                       mode='constant', constant_values=-1024)
    
    sCT2_data = np.pad(sCT2_data,
                       ((pad_width2[0], CT_bed_data.shape[0]-sCT2_data.shape[0]-pad_width2[0]),
                        (pad_width2[1], CT_bed_data.shape[1]-sCT2_data.shape[1]-pad_width2[1]),
                        (0, 0)),
                       mode='constant', constant_values=-1024)

    print(f"  After padding - CT_bed: {CT_bed_data.shape}, sCT1: {sCT1_data.shape}, sCT2: {sCT2_data.shape}")

    # Create body contour arrays
    sCT1_body_contour = np.zeros_like(sCT1_data)
    sCT2_body_contour = np.zeros_like(sCT2_data)
    
    print(f"  Processing {CT_bed_data.shape[2]} slices for body contours...")
    # Process each z-slice
    for z in range(CT_bed_data.shape[2]):
        body_mask = CT_bed_data[:,:,z] > body_contour_HU_th
        filled_mask = binary_fill_holes(body_mask)
        sCT1_body_contour[:,:,z] = filled_mask
        sCT2_body_contour[:,:,z] = filled_mask

    print(f"  Saving body contour masks...")
    # Save body contour masks
    sCT1_contour_nifti = nib.Nifti1Image(sCT1_body_contour, sCT1_nifti.affine, sCT1_nifti.header)
    sCT2_contour_nifti = nib.Nifti1Image(sCT2_body_contour, sCT2_nifti.affine, sCT2_nifti.header)

    sCT1_contour_path = f"{sCT1_src_folder}/sCT1_{case_name}_v2_body_contour.nii.gz"
    sCT2_contour_path = f"{sCT2_src_folder}/sCT2_{case_name}_v2_body_contour.nii.gz"

    nib.save(sCT1_contour_nifti, sCT1_contour_path)
    nib.save(sCT2_contour_nifti, sCT2_contour_path)

    print(f"  Creating masked bed data...")
    # Create masked versions of CT bed data
    sCT1_v3_bed = CT_bed_data.copy()
    sCT2_v3_bed = CT_bed_data.copy()

    # Replace CT bed data with sCT data where body contour is True
    sCT1_v3_bed[sCT1_body_contour == 1] = sCT1_data[sCT1_body_contour == 1]
    sCT2_v3_bed[sCT2_body_contour == 1] = sCT2_data[sCT2_body_contour == 1]

    print(f"  Saving masked bed data...")
    # Save masked bed data
    sCT1_v3_bed_nifti = nib.Nifti1Image(sCT1_v3_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    sCT2_v3_bed_nifti = nib.Nifti1Image(sCT2_v3_bed, CT_bed_nifti.affine, CT_bed_nifti.header)

    sCT1_v3_bed_path = f"{sCT1_src_folder}/sCT1_{case_name}_v3_bed.nii.gz"
    sCT2_v3_bed_path = f"{sCT2_src_folder}/sCT2_{case_name}_v3_bed.nii.gz"

    nib.save(sCT1_v3_bed_nifti, sCT1_v3_bed_path)
    nib.save(sCT2_v3_bed_nifti, sCT2_v3_bed_path)
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")

    
