import glob
import os
import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import binary_fill_holes, gaussian_filter
from tqdm import tqdm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process synthetic CT images and blend with CT bed data.')
    
    # Add arguments
    parser.add_argument('--input_dir', type=str, default="ErasmusMC/HUadj_resampled", 
                        help='Directory containing resampled synthetic CT images')
    parser.add_argument('--output_dir', type=str, default="ErasmusMC/HUadj_resampled_bed", 
                        help='Directory to save bed-recovered images')
    parser.add_argument('--CT_bed_folder', type=str, default="CTAC_bed/", 
                        help='CT bed folder')
    parser.add_argument('--contour_type', type=str, default="bcC", 
                        help='Body contour type (default: bcC)')
    parser.add_argument('--HU_threshold', type=int, default=-500, 
                        help='HU threshold for body contour')
    parser.add_argument('--blur_sigma', type=float, default=1.0, 
                        help='Sigma for Gaussian blur')
    parser.add_argument('--case_id', type=str, default=None, 
                        help='Process specific case ID (default: process all cases)')
    
    # Parse arguments
    args = parser.parse_args()
    
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
    
    # If a specific case ID is provided, only process that case
    if args.case_id:
        case_name_list = [args.case_id]
        print(f"Processing single case: {args.case_id}")
    else:
        print(f"Processing all {len(case_name_list)} cases from the predefined list")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Print processing information
    print(f"Starting processing with the following configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  CT bed folder: {args.CT_bed_folder}")
    print(f"  Contour type: {args.contour_type}")
    print(f"  HU threshold: {args.HU_threshold}")
    print(f"  Blur sigma: {args.blur_sigma}")

    for case_name in tqdm(case_name_list, desc="Processing cases"):
        process_case(case_name, args)

    print("Processing complete!")

def process_case(case_name, args):
    """Process a single case to recover the bed"""
    
    # Define file paths
    synct_path = os.path.join(args.input_dir, f"SynCT_{case_name}_{args.contour_type}_HUadj_resampled.nii.gz")
    output_base = os.path.join(args.output_dir, f"SynCT_{case_name}_{args.contour_type}")
    
    # Check if input file exists
    if not os.path.exists(synct_path):
        print(f"  Warning: Input file not found: {synct_path}")
        return
    
    # Find reference CT file for this case
    # Remove 'E' from case name to match reference file pattern
    case_id = case_name[1:]
    reference_files = glob.glob(f"{args.CT_bed_folder}/*_{case_id}_*.nii*")
    
    if not reference_files:
        print(f"  Warning: No reference file found for case {case_name}")
        return
    
    CT_bed_path = sorted(reference_files)[0]
    
    # Load synthetic CT data
    synct_nifti = nib.load(synct_path)
    synct_data = synct_nifti.get_fdata()
    
    # Load CT bed data
    CT_bed_nifti = nib.load(CT_bed_path)
    CT_bed_data = CT_bed_nifti.get_fdata()
    
    # Create body contour arrays
    synct_contour = np.zeros_like(synct_data, dtype=bool)
    CT_bed_contour = np.zeros_like(CT_bed_data, dtype=bool)
    
    # Process each z-slice for body contours
    for z in range(synct_data.shape[2]):
        # Create mask from synthetic CT data using HU threshold
        synct_mask = synct_data[:,:,z] > args.HU_threshold
        # Fill holes in the mask
        synct_filled_mask = binary_fill_holes(synct_mask)
        # Save the filled mask
        synct_contour[:,:,z] = synct_filled_mask
        
        # Create mask from CT bed data using HU threshold
        CT_bed_mask = CT_bed_data[:,:,z] > args.HU_threshold
        # Fill holes in the mask
        CT_bed_filled_mask = binary_fill_holes(CT_bed_mask)
        # Save the filled mask
        CT_bed_contour[:,:,z] = CT_bed_filled_mask
    
    # Take the intersection of the contours
    intersection_contour = synct_contour & CT_bed_contour
    
    # Convert boolean mask to float for blurring
    intersection_float = intersection_contour.astype(np.float32)
    
    # Apply Gaussian blur to create soft edges
    blurred_contour = gaussian_filter(intersection_float, sigma=args.blur_sigma)
    
    # Save intersection body contour mask (for debugging/visualization)
    intersection_nifti = nib.Nifti1Image(intersection_contour.astype(np.int16), 
                                        synct_nifti.affine, synct_nifti.header)
    
    intersection_path = f"{output_base}_intersection_contour.nii.gz"
    nib.save(intersection_nifti, intersection_path)
    
    # Save blurred contour for visualization/debugging
    blurred_nifti = nib.Nifti1Image(blurred_contour, synct_nifti.affine, synct_nifti.header)
    blurred_path = f"{output_base}_blurred_contour.nii.gz"
    nib.save(blurred_nifti, blurred_path)
    
    # Process the entire volume using the blurred contour as weight
    synct_with_bed = (blurred_contour * synct_data + (1 - blurred_contour) * CT_bed_data)
    
    # Save the final blended image
    synct_with_bed_nifti = nib.Nifti1Image(synct_with_bed, CT_bed_nifti.affine, CT_bed_nifti.header)
    synct_with_bed_path = f"{output_base}_HUadj_resampled_bed.nii.gz"
    nib.save(synct_with_bed_nifti, synct_with_bed_path)

if __name__ == "__main__":
    main()
