import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure, iterate_structure, median_filter

case_name_list = [
    # 'E4058',
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
]

folder_name = "NAC_CTAC_Spacing15"
output_folder = "NAC_body_contour_thresholds"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Fixed threshold value
threshold = 500

# Fixed sigma value for Gaussian blur
sigma = 1.0

# Parameters for dilation and erosion
morph_iterations = 3

# Create 3D structuring elements for better connectivity
# 2D structuring element for slice processing
basic_struct_2d = generate_binary_structure(2, 2)  # 8-connectivity in 2D
struct_element_2d = iterate_structure(basic_struct_2d, 3)  # 7x7 kernel

# 3D structuring element for volume processing
basic_struct_3d = generate_binary_structure(3, 1)  # 6-connectivity in 3D
small_struct_3d = basic_struct_3d  # Basic 3x3x3 with 6-connectivity

for case_name in case_name_list:
    print(f"Processing case: {case_name}")
    NAC_path = f"{folder_name}/NAC_{case_name}_256.nii.gz"
    
    # Load NAC data
    print(f"  Loading NAC data from: {NAC_path}")
    NAC_nifti = nib.load(NAC_path)
    NAC_data = NAC_nifti.get_fdata()
    
    print(f"  NAC data shape: {NAC_data.shape}")
    
    # Apply Gaussian blurring to the entire volume
    print(f"  Applying Gaussian blur with sigma={sigma}...")
    NAC_blurred = gaussian_filter(NAC_data, sigma=sigma)
    
    # Create initial threshold mask
    print(f"  Creating initial threshold mask with threshold={threshold}...")
    initial_mask = NAC_blurred > threshold
    
    # Process in each dimension
    print(f"  Processing contours in each dimension...")
    
    # Initialize contours for each dimension
    x_contour = np.zeros_like(initial_mask, dtype=bool)
    y_contour = np.zeros_like(initial_mask, dtype=bool)
    z_contour = np.zeros_like(initial_mask, dtype=bool)
    
    # Process along Z dimension (axial slices)
    print(f"  Processing Z dimension (axial slices)...")
    for z in range(NAC_data.shape[2]):
        # Get the slice
        slice_mask = initial_mask[:,:,z]
        
        # Dilate then erode (close operation) with equal iterations
        dilated_mask = binary_dilation(slice_mask, structure=struct_element_2d, iterations=morph_iterations)
        eroded_mask = binary_erosion(dilated_mask, structure=struct_element_2d, iterations=morph_iterations)
        
        # Fill holes after morphological operations
        final_mask = binary_fill_holes(eroded_mask)
        
        # Save to z_contour
        z_contour[:,:,z] = final_mask
    
    # Process along Y dimension (coronal slices)
    print(f"  Processing Y dimension (coronal slices)...")
    for y in range(NAC_data.shape[1]):
        # Get the slice
        slice_mask = initial_mask[:,y,:]
        
        # Dilate then erode (close operation) with equal iterations
        dilated_mask = binary_dilation(slice_mask, structure=struct_element_2d, iterations=morph_iterations)
        eroded_mask = binary_erosion(dilated_mask, structure=struct_element_2d, iterations=morph_iterations)
        
        # Fill holes after morphological operations
        final_mask = binary_fill_holes(eroded_mask)
        
        # Save to y_contour
        y_contour[:,y,:] = final_mask
    
    # Process along X dimension (sagittal slices)
    print(f"  Processing X dimension (sagittal slices)...")
    for x in range(NAC_data.shape[0]):
        # Get the slice
        slice_mask = initial_mask[x,:,:]
        
        # Dilate then erode (close operation) with equal iterations
        dilated_mask = binary_dilation(slice_mask, structure=struct_element_2d, iterations=morph_iterations)
        eroded_mask = binary_erosion(dilated_mask, structure=struct_element_2d, iterations=morph_iterations)
        
        # Fill holes after morphological operations
        final_mask = binary_fill_holes(eroded_mask)
        
        # Save to x_contour
        x_contour[x,:,:] = final_mask
    
    # Create union contour (logical OR)
    print(f"  Creating union contour...")
    union_contour = x_contour | y_contour | z_contour
    
    # Additional processing for the union contour to remove edge artifacts (毛刺)
    print(f"  Performing additional processing to remove edge artifacts...")
    
    # 1. Apply 3D median filter to remove small spikes
    print(f"    Applying 3D median filter...")
    union_filtered = median_filter(union_contour.astype(np.uint8), size=3)
    
    # 2. Small erosion followed by dilation to remove thin spikes (using 3D structuring element)
    print(f"    Applying small erosion and dilation to remove thin spikes...")
    union_eroded = binary_erosion(union_filtered, structure=small_struct_3d, iterations=1)
    union_refined = binary_dilation(union_eroded, structure=small_struct_3d, iterations=1)
    
    # 3. Fill holes in each dimension again
    print(f"    Filling holes in each dimension...")
    refined_contour = np.copy(union_refined).astype(bool)
    
    # Fill holes in Z dimension
    for z in range(NAC_data.shape[2]):
        refined_contour[:,:,z] = binary_fill_holes(refined_contour[:,:,z])
    
    # Fill holes in Y dimension
    for y in range(NAC_data.shape[1]):
        refined_contour[:,y,:] = binary_fill_holes(refined_contour[:,y,:])
    
    # Fill holes in X dimension
    for x in range(NAC_data.shape[0]):
        refined_contour[x,:,:] = binary_fill_holes(refined_contour[x,:,:])
    
    # 4. Final smoothing with small closing operation (using 3D structuring element)
    print(f"    Performing final smoothing...")
    final_contour = binary_dilation(refined_contour, structure=small_struct_3d, iterations=1)
    final_contour = binary_erosion(final_contour, structure=small_struct_3d, iterations=1)
    
    # Save the original contours
    print(f"  Saving original dimension contours...")
    x_contour_nifti = nib.Nifti1Image(x_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    y_contour_nifti = nib.Nifti1Image(y_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    z_contour_nifti = nib.Nifti1Image(z_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    
    nib.save(x_contour_nifti, f"{output_folder}/NAC_{case_name}_x_contour.nii.gz")
    nib.save(y_contour_nifti, f"{output_folder}/NAC_{case_name}_y_contour.nii.gz")
    nib.save(z_contour_nifti, f"{output_folder}/NAC_{case_name}_z_contour.nii.gz")
    
    # Save the union and refined contours
    print(f"  Saving union and refined contours...")
    union_nifti = nib.Nifti1Image(union_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    refined_nifti = nib.Nifti1Image(refined_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    final_nifti = nib.Nifti1Image(final_contour.astype(np.int16), NAC_nifti.affine, NAC_nifti.header)
    
    nib.save(union_nifti, f"{output_folder}/NAC_{case_name}_union_contour.nii.gz")
    nib.save(refined_nifti, f"{output_folder}/NAC_{case_name}_refined_contour.nii.gz")
    nib.save(final_nifti, f"{output_folder}/NAC_{case_name}_final_contour.nii.gz")
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")
