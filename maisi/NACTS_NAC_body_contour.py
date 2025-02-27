import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, binary_dilation, binary_erosion

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

folder_name = "NAC_CTAC_Spacing15"
output_folder = "NAC_body_contour_thresholds"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Fixed threshold value
threshold = 500

# Sigma values for Gaussian blur to try
sigma_values = [0.5, 1.0, 1.5, 2.0]

# Parameters for dilation and erosion
dilation_iterations = 3
erosion_iterations = 2

for case_name in case_name_list:
    print(f"Processing case: {case_name}")
    NAC_path = f"{folder_name}/NAC_{case_name}_256.nii.gz"
    
    # Load NAC data
    print(f"  Loading NAC data from: {NAC_path}")
    NAC_nifti = nib.load(NAC_path)
    NAC_data = NAC_nifti.get_fdata()
    
    print(f"  NAC data shape: {NAC_data.shape}")
    
    for sigma in sigma_values:
        print(f"  Processing with sigma={sigma}")
        
        # Apply Gaussian blurring to the entire volume
        print(f"    Applying Gaussian blur...")
        NAC_blurred = gaussian_filter(NAC_data, sigma=sigma)
        
        # Create initial threshold mask
        print(f"    Creating initial threshold mask with threshold={threshold}...")
        initial_mask = NAC_blurred > threshold
        
        # Process in each dimension
        print(f"    Processing contours in each dimension...")
        
        # Initialize contours for each dimension
        x_contour = np.zeros_like(initial_mask, dtype=bool)
        y_contour = np.zeros_like(initial_mask, dtype=bool)
        z_contour = np.zeros_like(initial_mask, dtype=bool)
        
        # Process along Z dimension (axial slices)
        print(f"    Processing Z dimension (axial slices)...")
        for z in range(NAC_data.shape[2]):
            # Get the slice
            slice_mask = initial_mask[:,:,z]
            
            # Dilate then erode (close operation)
            dilated_mask = binary_dilation(slice_mask, iterations=dilation_iterations)
            eroded_mask = binary_erosion(dilated_mask, iterations=erosion_iterations)
            
            # Fill holes after morphological operations
            final_mask = binary_fill_holes(eroded_mask)
            
            # Save to z_contour
            z_contour[:,:,z] = final_mask
        
        # Process along Y dimension (coronal slices)
        print(f"    Processing Y dimension (coronal slices)...")
        for y in range(NAC_data.shape[1]):
            # Get the slice
            slice_mask = initial_mask[:,y,:]
            
            # Dilate then erode (close operation)
            dilated_mask = binary_dilation(slice_mask, iterations=dilation_iterations)
            eroded_mask = binary_erosion(dilated_mask, iterations=erosion_iterations)
            
            # Fill holes after morphological operations
            final_mask = binary_fill_holes(eroded_mask)
            
            # Save to y_contour
            y_contour[:,y,:] = final_mask
        
        # Process along X dimension (sagittal slices)
        print(f"    Processing X dimension (sagittal slices)...")
        for x in range(NAC_data.shape[0]):
            # Get the slice
            slice_mask = initial_mask[x,:,:]
            
            # Dilate then erode (close operation)
            dilated_mask = binary_dilation(slice_mask, iterations=dilation_iterations)
            eroded_mask = binary_erosion(dilated_mask, iterations=erosion_iterations)
            
            # Fill holes after morphological operations
            final_mask = binary_fill_holes(eroded_mask)
            
            # Save to x_contour
            x_contour[x,:,:] = final_mask
        
        # Combine contours using different methods
        print(f"    Combining contours using different methods...")
        
        # Intersection (logical AND)
        intersection_contour = x_contour & y_contour & z_contour
        
        # Union (logical OR)
        union_contour = x_contour | y_contour | z_contour
        
        # Majority vote (at least 2 out of 3)
        vote_sum = x_contour.astype(np.uint8) + y_contour.astype(np.uint8) + z_contour.astype(np.uint8)
        majority_contour = vote_sum >= 2
        
        # Save individual dimension contours
        print(f"    Saving individual dimension contours...")
        
        x_contour_nifti = nib.Nifti1Image(x_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        y_contour_nifti = nib.Nifti1Image(y_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        z_contour_nifti = nib.Nifti1Image(z_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        
        nib.save(x_contour_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_x_contour.nii.gz")
        nib.save(y_contour_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_y_contour.nii.gz")
        nib.save(z_contour_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_z_contour.nii.gz")
        
        # Save combined contours
        print(f"    Saving combined contours...")
        
        intersection_nifti = nib.Nifti1Image(intersection_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        union_nifti = nib.Nifti1Image(union_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        majority_nifti = nib.Nifti1Image(majority_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
        
        nib.save(intersection_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_intersection_contour.nii.gz")
        nib.save(union_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_union_contour.nii.gz")
        nib.save(majority_nifti, f"{output_folder}/NAC_{case_name}_sigma{sigma}_majority_contour.nii.gz")
        
        print(f"  Completed processing for sigma={sigma}")
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")
