import glob
import os
import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import binary_fill_holes, gaussian_filter

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process sCT images and blend with CT bed data.')
    
    # Add arguments
    parser.add_argument('--sCT0', action='store_true', help='Process sCT0 images')
    parser.add_argument('--sCT1', action='store_true', help='Process sCT1 images')
    parser.add_argument('--sCT2', action='store_true', help='Process sCT2 images')
    parser.add_argument('--all', action='store_true', help='Process all sCT types (sCT0, sCT1, sCT2)')
    parser.add_argument('--sCT0_folder', type=str, default="James_36/sCT0_MLAA", help='sCT0 source folder')
    parser.add_argument('--sCT1_folder', type=str, default="James_36/sCT1_MLAA", help='sCT1 source folder')
    parser.add_argument('--sCT2_folder', type=str, default="James_36/sCT2_MLAA", help='sCT2 source folder')
    parser.add_argument('--CT_bed_folder', type=str, default="CTAC_bed/", help='CT bed folder')
    parser.add_argument('--HU_threshold', type=int, default=-500, help='HU threshold for body contour')
    parser.add_argument('--blur_sigma', type=float, default=1.0, help='Sigma for Gaussian blur')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no specific sCT type is selected and --all is not used, default to processing all types
    if not (args.sCT0 or args.sCT1 or args.sCT2 or args.all):
        args.all = True
    
    # Determine which sCT types to process
    process_sCT0 = args.sCT0 or args.all
    process_sCT1 = args.sCT1 or args.all
    process_sCT2 = args.sCT2 or args.all
    
    # List of cases to process
    case_name_list = [
        'E4058',
        # 'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
        # 'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
        # 'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
        # 'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
        # 'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
        # 'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
        # 'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
        # 'E4139', 
    ]

    # Define input/output directories
    sCT0_src_folder = args.sCT0_folder
    sCT1_src_folder = args.sCT1_folder
    sCT2_src_folder = args.sCT2_folder
    CT_bed_folder = args.CT_bed_folder

    # HU threshold for body contour
    body_contour_HU_th = args.HU_threshold

    # Edge blurring parameters
    blur_sigma = args.blur_sigma  # Sigma for Gaussian blur (higher = more blurring)

    # Print processing information
    print(f"Starting processing {len(case_name_list)} cases")
    print(f"Processing types: {'sCT0 ' if process_sCT0 else ''}{'sCT1 ' if process_sCT1 else ''}{'sCT2' if process_sCT2 else ''}")
    print(f"HU threshold: {body_contour_HU_th}, Blur sigma: {blur_sigma}")

    for idx, case_name in enumerate(case_name_list, 1):
        print(f"[{idx}/{len(case_name_list)}] Processing case: {case_name}")

        # Load CT bed data (needed for all sCT types)
        CT_bed_path = sorted(glob.glob(f"{CT_bed_folder}*_{case_name[1:]}_*.nii"))[0]
        print(f"  [{idx}/{len(case_name_list)}] Loading CT bed data...")
        CT_bed_nifti = nib.load(CT_bed_path)
        CT_bed_data = CT_bed_nifti.get_fdata()
        CT_bed_contour = np.zeros_like(CT_bed_data, dtype=bool)
        
        # Process CT bed contour (needed for all sCT types)
        print(f"  [{idx}/{len(case_name_list)}] Processing CT bed contour...")
        for z in range(CT_bed_data.shape[2]):
            CT_bed_mask = CT_bed_data[:,:,z] > body_contour_HU_th
            CT_bed_contour[:,:,z] = binary_fill_holes(CT_bed_mask)
        
        # Process sCT0 if selected
        if process_sCT0:
            process_sCT(idx, case_name, "sCT0", sCT0_src_folder, CT_bed_nifti, CT_bed_data, 
                       CT_bed_contour, body_contour_HU_th, blur_sigma, len(case_name_list))
        
        # Process sCT1 if selected
        if process_sCT1:
            process_sCT(idx, case_name, "sCT1", sCT1_src_folder, CT_bed_nifti, CT_bed_data, 
                       CT_bed_contour, body_contour_HU_th, blur_sigma, len(case_name_list))
        
        # Process sCT2 if selected
        if process_sCT2:
            process_sCT(idx, case_name, "sCT2", sCT2_src_folder, CT_bed_nifti, CT_bed_data, 
                       CT_bed_contour, body_contour_HU_th, blur_sigma, len(case_name_list))
        
        print(f"Completed case: {case_name}\n")

    print("Processing complete!")

def process_sCT(idx, case_name, sCT_type, src_folder, CT_bed_nifti, CT_bed_data, 
               CT_bed_contour, body_contour_HU_th, blur_sigma, total_cases):
    """Process a single sCT type for a given case"""
    
    print(f"  [{idx}/{total_cases}] Processing {sCT_type} for case {case_name}...")
    
    # Load sCT data
    sCT_src_path = f"{src_folder}/{sCT_type}_{case_name}_v1_3dresample.nii.gz"
    sCT_nifti = nib.load(sCT_src_path)
    sCT_data = sCT_nifti.get_fdata()
    
    print(f"  {sCT_type} data shape: {sCT_data.shape}, CT_bed: {CT_bed_data.shape}")
    
    # Pad sCT data to match CT_bed dimensions
    pad_width = ((CT_bed_data.shape[0] - sCT_data.shape[0])//2,
                 (CT_bed_data.shape[1] - sCT_data.shape[1])//2)
    
    sCT_data = np.pad(sCT_data, 
                      ((pad_width[0], CT_bed_data.shape[0]-sCT_data.shape[0]-pad_width[0]),
                       (pad_width[1], CT_bed_data.shape[1]-sCT_data.shape[1]-pad_width[1]),
                       (0, 0)), 
                      mode='constant', constant_values=-1024)
    
    print(f"  After padding - {sCT_type}: {sCT_data.shape}")
    
    # Create body contour array
    sCT_contour = np.zeros_like(sCT_data, dtype=bool)
    
    # Process each z-slice for body contour
    for z in range(sCT_data.shape[2]):
        # Create mask from sCT data using HU threshold
        sCT_mask = sCT_data[:,:,z] > body_contour_HU_th
        # Fill holes in the mask
        sCT_filled_mask = binary_fill_holes(sCT_mask)
        # Save the filled mask
        sCT_contour[:,:,z] = sCT_filled_mask
    
    # Take the intersection of the contours
    intersection_contour = sCT_contour & CT_bed_contour
    
    # Convert boolean mask to float for blurring
    intersection_float = intersection_contour.astype(np.float32)
    
    # Apply Gaussian blur to create soft edges
    blurred_contour = gaussian_filter(intersection_float, sigma=blur_sigma)
    
    # Save intersection body contour mask
    intersection_nifti = nib.Nifti1Image(intersection_contour.astype(np.int16), 
                                        sCT_nifti.affine, sCT_nifti.header)
    
    intersection_path = f"{src_folder}/{sCT_type}_{case_name}_v2_intersection_contour.nii.gz"
    nib.save(intersection_nifti, intersection_path)
    
    # Save blurred contour for visualization/debugging
    blurred_nifti = nib.Nifti1Image(blurred_contour, sCT_nifti.affine, sCT_nifti.header)
    blurred_path = f"{src_folder}/{sCT_type}_{case_name}_v2_blurred_contour.nii.gz"
    nib.save(blurred_nifti, blurred_path)
    
    # Process the entire volume using the blurred contour as weight
    sCT_v4_bed = (blurred_contour * sCT_data + (1 - blurred_contour) * CT_bed_data)
    
    # Save masked bed data
    sCT_v4_bed_nifti = nib.Nifti1Image(sCT_v4_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    sCT_v4_bed_path = f"{src_folder}/{sCT_type}_{case_name}_v4_bed_blurred_edge.nii.gz"
    nib.save(sCT_v4_bed_nifti, sCT_v4_bed_path)
    
    print(f"  Completed processing {sCT_type} for case {case_name}")

if __name__ == "__main__":
    main()

    
