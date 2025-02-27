import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

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

# Different thresholds to try
thresholds = [250, 500, 750, 1000, 1250]

# Different sigma values to try
sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0]

for case_name in case_name_list:
    print(f"Processing case: {case_name}")
    NAC_path = f"{folder_name}/NAC_{case_name}_256.nii.gz"
    
    # Load NAC data
    print(f"  Loading NAC data from: {NAC_path}")
    NAC_nifti = nib.load(NAC_path)
    NAC_data = NAC_nifti.get_fdata()
    
    print(f"  NAC data shape: {NAC_data.shape}")
    
    # Process each sigma value
    for sigma in sigma_values:
        print(f"  Processing with sigma={sigma}")
        
        # Apply Gaussian blurring to the entire volume
        print(f"    Applying Gaussian blur...")
        NAC_blurred = gaussian_filter(NAC_data, sigma=sigma)
        
        # Save blurred NAC data for reference
        blurred_path = f"{output_folder}/NAC_{case_name}_sigma{sigma}_blurred.nii.gz"
        blurred_nifti = nib.Nifti1Image(NAC_blurred, NAC_nifti.affine, NAC_nifti.header)
        nib.save(blurred_nifti, blurred_path)
        print(f"    Saved blurred NAC data to: {blurred_path}")
        
        # Process each threshold
        for threshold in thresholds:
            print(f"    Processing threshold: {threshold}")
            
            # Create empty contour array
            contour = np.zeros_like(NAC_data, dtype=bool)
            
            # Process each z-slice
            for z in range(NAC_data.shape[2]):
                # Create mask using threshold on blurred data
                mask = NAC_blurred[:,:,z] > threshold
                
                # Fill holes in the mask
                filled_mask = binary_fill_holes(mask)
                
                # Save the filled mask
                contour[:,:,z] = filled_mask
            
            # Save the contour mask
            output_path = f"{output_folder}/NAC_{case_name}_sigma{sigma}_th{threshold}.nii.gz"
            contour_nifti = nib.Nifti1Image(contour.astype(np.int16), 
                                            NAC_nifti.affine, 
                                            NAC_nifti.header)
            
            print(f"    Saving contour to: {output_path}")
            nib.save(contour_nifti, output_path)
        
        print(f"  Completed processing for sigma={sigma}")
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!")
