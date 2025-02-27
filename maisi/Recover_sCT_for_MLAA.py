import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

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

# Edge blurring parameters
blur_sigma = 1.0  # Sigma for Gaussian blur (higher = more blurring)

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

    # Create body contour arrays for both CT_bed and sCT data
    sCT1_contour = np.zeros_like(sCT1_data, dtype=bool)
    sCT2_contour = np.zeros_like(sCT2_data, dtype=bool)
    CT_bed_contour = np.zeros_like(CT_bed_data, dtype=bool)
    
    print(f"  [{idx}/{len(case_name_list)}] Processing {CT_bed_data.shape[2]} slices for body contours...")
    # Process each z-slice
    for z in range(CT_bed_data.shape[2]):
        # Create masks from sCT data using HU threshold
        sCT1_mask = sCT1_data[:,:,z] > body_contour_HU_th
        sCT2_mask = sCT2_data[:,:,z] > body_contour_HU_th
        CT_bed_mask = CT_bed_data[:,:,z] > body_contour_HU_th
        
        # Fill holes in the masks
        sCT1_filled_mask = binary_fill_holes(sCT1_mask)
        sCT2_filled_mask = binary_fill_holes(sCT2_mask)
        CT_bed_filled_mask = binary_fill_holes(CT_bed_mask)
        
        # Save the filled masks
        sCT1_contour[:,:,z] = sCT1_filled_mask
        sCT2_contour[:,:,z] = sCT2_filled_mask
        CT_bed_contour[:,:,z] = CT_bed_filled_mask

    # Take the intersection of the contours
    intersection_contour1 = sCT1_contour & CT_bed_contour
    intersection_contour2 = sCT2_contour & CT_bed_contour
    
    # Create blurred versions of the intersection contours for edge blending
    print(f"  Creating blurred edge masks...")
    # Convert boolean masks to float for blurring
    intersection_float1 = intersection_contour1.astype(np.float32)
    intersection_float2 = intersection_contour2.astype(np.float32)
    
    # Apply Gaussian blur to create soft edges
    blurred_contour1 = gaussian_filter(intersection_float1, sigma=blur_sigma)
    blurred_contour2 = gaussian_filter(intersection_float2, sigma=blur_sigma)
    
    print(f"  Saving intersection body contour masks...")
    # Save intersection body contour masks
    intersection_nifti1 = nib.Nifti1Image(intersection_contour1.astype(np.int16), 
                                         sCT1_nifti.affine, sCT1_nifti.header)
    intersection_nifti2 = nib.Nifti1Image(intersection_contour2.astype(np.int16), 
                                         sCT2_nifti.affine, sCT2_nifti.header)
    
    intersection_path1 = f"{sCT1_src_folder}/sCT1_{case_name}_v2_intersection_contour.nii.gz"
    intersection_path2 = f"{sCT2_src_folder}/sCT2_{case_name}_v2_intersection_contour.nii.gz"
    
    nib.save(intersection_nifti1, intersection_path1)
    nib.save(intersection_nifti2, intersection_path2)
    
    # Save blurred contours for visualization/debugging
    blurred_nifti1 = nib.Nifti1Image(blurred_contour1, sCT1_nifti.affine, sCT1_nifti.header)
    blurred_nifti2 = nib.Nifti1Image(blurred_contour2, sCT2_nifti.affine, sCT2_nifti.header)
    
    blurred_path1 = f"{sCT1_src_folder}/sCT1_{case_name}_v2_blurred_contour.nii.gz"
    blurred_path2 = f"{sCT2_src_folder}/sCT2_{case_name}_v2_blurred_contour.nii.gz"
    
    nib.save(blurred_nifti1, blurred_path1)
    nib.save(blurred_nifti2, blurred_path2)

    print(f"  Creating masked bed data with blurred edges...")
    # Memory-efficient blending approach
    print(f"  Applying blending with blurred edges...")
    
    # Process the entire volumes directly using the blurred contours as weights
    sCT1_v4_bed = (blurred_contour1 * sCT1_data + 
                  (1 - blurred_contour1) * CT_bed_data)
    
    sCT2_v4_bed = (blurred_contour2 * sCT2_data + 
                  (1 - blurred_contour2) * CT_bed_data)

    print(f"  Saving masked bed data with blurred edges...")
    # Save masked bed data
    sCT1_v4_bed_nifti = nib.Nifti1Image(sCT1_v4_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    sCT2_v4_bed_nifti = nib.Nifti1Image(sCT2_v4_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    
    sCT1_v4_bed_path = f"{sCT1_src_folder}/sCT1_{case_name}_v4_bed_blurred_edge.nii.gz"
    sCT2_v4_bed_path = f"{sCT2_src_folder}/sCT2_{case_name}_v4_bed_blurred_edge.nii.gz"
    
    nib.save(sCT1_v4_bed_nifti, sCT1_v4_bed_path)
    nib.save(sCT2_v4_bed_nifti, sCT2_v4_bed_path)
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")

    
