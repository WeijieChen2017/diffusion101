# nii2dcm nifti-file.nii.gz dicom-output-directory/ --dicom-type CT -r reference-dicom-file.dcm
# wxc321@l-mimrtl02539:/shares/mimrtl/Users/James/'synthetic CT PET AC'$

# case_name_list = [
#     'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
#     'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
#     'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
#     'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
#     'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
#     'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
#     'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
#     'E4139', 
# ]
case_name_list = [
    'E4058',
]

import os
import glob
import nibabel as nib
import pydicom
import numpy as np
from pathlib import Path

nii_folder = "sCT_NAC"

def convert_nifti_to_dicom(nii_file, dicom_output_directory, reference_dicom_folder):
    """
    Convert a NIfTI file to DICOM series while preserving original DICOM metadata.
    
    Args:
        nii_file (str): Path to input NIfTI file
        dicom_output_directory (str): Path to output DICOM directory
        reference_dicom_folder (str): Path to reference DICOM folder
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(dicom_output_directory, exist_ok=True)
        
        # Load NIfTI data
        nii_img = nib.load(nii_file)
        nii_data = nii_img.get_fdata()
        
        # Get list of reference DICOM files
        ref_dicom_files = sorted(glob.glob(os.path.join(reference_dicom_folder, '*.dcm')))
        
        if len(ref_dicom_files) == 0:
            print(f"Error: No DICOM files found in {reference_dicom_folder}")
            return False
            
        # Check if dimensions match
        if len(ref_dicom_files) != nii_data.shape[2]:
            print(f"Error: Number of DICOM slices ({len(ref_dicom_files)}) "
                  f"doesn't match NIfTI z-dimension ({nii_data.shape[2]})")
            return False
        
        # Process each slice
        for slice_idx in range(nii_data.shape[2]):
            # Load corresponding reference DICOM
            ref_dcm = pydicom.dcmread(ref_dicom_files[slice_idx])
            
            # Extract and process slice from NIfTI
            slice_data = nii_data[:, :, slice_idx]
            slice_data = slice_data.astype(np.int16)
            
            # Verify dimensions match
            if (slice_data.shape[0] != ref_dcm.Rows or 
                slice_data.shape[1] != ref_dcm.Columns):
                print(f"Error: Slice dimensions don't match for slice {slice_idx}")
                return False
            
            # Update DICOM with new pixel data
            ref_dcm.PixelData = slice_data.tobytes()
            
            # Save modified DICOM
            output_path = os.path.join(dicom_output_directory, 
                                     os.path.basename(ref_dicom_files[slice_idx]))
            ref_dcm.save_as(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

# Main execution
for case_name in case_name_list:
    try:
        # Get the NIfTI file path
        nii_file = sorted(glob.glob(f'Test_1sub_sCTnii/{nii_folder}/*{case_name}*.nii.gz'))[0]
        
        # Set up paths
        dicom_output_directory = f'Test_1sub_sCTdcm/{nii_folder}/PETLYMPH_{case_name[1:]}'
        reference_dicom_folder = f'Duetto_Output_B100/PETLYMPH_{case_name[1:]}'
        
        # Perform conversion
        if convert_nifti_to_dicom(nii_file, dicom_output_directory, reference_dicom_folder):
            print(f"Successfully converted {case_name} - {nii_file} to DICOM")
        else:
            print(f"Failed to convert {case_name}")
            
    except Exception as e:
        print(f"Error processing case {case_name}: {str(e)}")
        continue
    