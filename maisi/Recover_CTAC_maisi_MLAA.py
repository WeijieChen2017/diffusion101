import glob
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

# Define input/output directories
CTAC_maisi_src_folder = "James_36/CTAC_maisi"
CTAC_maisi_dst_folder = "James_36/CTAC_maisi"
CT_bed_folder = "CTAC_bed/"

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

# Edge blurring parameters
blur_sigma = 1.0  # Sigma for Gaussian blur (higher = more blurring)
edge_width = 5    # Width of the edge transition zone in pixels

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

    # Create body contour arrays for both CT_bed and CTAC_maisi
    CTAC_maisi_contour = np.zeros_like(CTAC_maisi_data, dtype=bool)
    CT_bed_contour = np.zeros_like(CT_bed_data, dtype=bool)
    
    print(f"  [{idx}/{len(case_name_list)}] Processing {CT_bed_data.shape[2]} slices for body contours...")
    # Process each z-slice
    for z in range(CT_bed_data.shape[2]):
        # Create mask from CTAC data using HU threshold
        CTAC_maisi_mask = CTAC_maisi_data[:,:,z] > body_contour_HU_th
        # Create mask from CT_bed data using HU threshold
        CT_bed_mask = CT_bed_data[:,:,z] > body_contour_HU_th
        
        # Fill holes in both masks
        CTAC_maisi_filled_mask = binary_fill_holes(CTAC_maisi_mask)
        CT_bed_filled_mask = binary_fill_holes(CT_bed_mask)
        
        # Save the filled masks
        CTAC_maisi_contour[:,:,z] = CTAC_maisi_filled_mask
        CT_bed_contour[:,:,z] = CT_bed_filled_mask

    # Take the intersection of the two body contours
    intersection_contour = CTAC_maisi_contour & CT_bed_contour
    
    # Create a blurred version of the intersection contour for edge blending
    print(f"  Creating blurred edge mask...")
    # Convert boolean mask to float for blurring
    intersection_float = intersection_contour.astype(np.float32)
    
    # Apply Gaussian blur to create soft edges
    blurred_contour = gaussian_filter(intersection_float, sigma=blur_sigma)
    
    # Create edge mask (values between 0 and 1 at the edges)
    edge_mask = np.zeros_like(blurred_contour)
    edge_mask[(blurred_contour > 0) & (blurred_contour < 1)] = blurred_contour[(blurred_contour > 0) & (blurred_contour < 1)]
    
    # Normalize edge mask to range [0,1]
    if np.max(edge_mask) > 0:
        edge_mask = edge_mask / np.max(edge_mask)
    
    print(f"  Saving intersection body contour mask...")
    # Save intersection body contour mask
    intersection_nifti = nib.Nifti1Image(intersection_contour.astype(np.int16), 
                                        CTAC_maisi_nifti.affine, CTAC_maisi_nifti.header)
    intersection_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v2_intersection_contour.nii.gz"
    nib.save(intersection_nifti, intersection_path)
    
    # Save edge mask for visualization/debugging
    edge_mask_nifti = nib.Nifti1Image(edge_mask, CTAC_maisi_nifti.affine, CTAC_maisi_nifti.header)
    edge_mask_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v2_edge_mask.nii.gz"
    nib.save(edge_mask_nifti, edge_mask_path)

    print(f"  Creating masked bed data with blurred edges...")
    # Create masked version of CT bed data with blurred edges
    CTAC_maisi_v4_bed = CT_bed_data.copy()

    # Apply the core intersection mask (hard boundary)
    core_mask = blurred_contour > 0.9  # Values close to 1 are considered core
    CTAC_maisi_v4_bed[core_mask] = CTAC_maisi_data[core_mask]
    
    # Apply blending at the edges
    edge_region = (blurred_contour > 0) & (blurred_contour <= 0.9)
    blend_weights = blurred_contour[edge_region].reshape(-1, 1)
    
    # Blend the values at the edges
    CTAC_maisi_v4_bed[edge_region] = (
        blend_weights * CTAC_maisi_data[edge_region] + 
        (1 - blend_weights) * CT_bed_data[edge_region]
    )

    print(f"  Saving masked bed data with blurred edges...")
    # Save masked bed data
    CTAC_maisi_v4_bed_nifti = nib.Nifti1Image(CTAC_maisi_v4_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    CTAC_maisi_v4_bed_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v4_bed_blurred_edge.nii.gz"
    nib.save(CTAC_maisi_v4_bed_nifti, CTAC_maisi_v4_bed_path)
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")