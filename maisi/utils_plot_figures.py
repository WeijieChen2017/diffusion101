import nibabel as nib
import numpy as np
import os

case_name_list = [
    'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
]

for case_name in case_name_list:

    nac_pet_path = f"NAC_CTAC_Spacing15/NAC_{case_name}_256.nii.gz"
    ctac_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"
    ctac_seg_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_TS.nii.gz"
    ctac_body_con_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_TS_body.nii.gz"
    catc_seg_200_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_TS_label.nii.gz"
    ctac_recon_path = f"NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_{case_name}_TS_MAISI.nii.gz"
    ctac_recon_adjusted_path = f"NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_{case_name}_TS_MAISI_adjusted.nii.gz"
    sCT0_path = f"sCT0_LDM36/sCT0_{case_name}_400.nii.gz"
    sCT0_seg_path = f"James_36/overlap_seg/vanila_overlap_{case_name}_Spacing15.nii.gz"
    sCT1_path = f"James_36/synCT/SynCT_{case_name}.nii.gz"
    sCT1_adjusted_path = f"James_36/synCT/SynCT_{case_name}.nii.gz"
    sCT2_path = f"James_36/synCT/inference_20250128_noon/CTAC_{case_name}_TS_MAISI.nii.gz"
    sCT2_adjusted_path = f"James_36/synCT/inference_20250128_noon/CTAC_{case_name}_TS_MAISI_adjusted.nii.gz"
    
    try:
        # Load both nifti files
        sCT0_nifti = nib.load(sCT0_path)
        ctac_nifti = nib.load(ctac_path)
        
        # Create new nifti file with sCT0 data but CTAC header/affine
        aligned_nifti = nib.Nifti1Image(
            sCT0_nifti.get_fdata(),
            ctac_nifti.affine,
            header=ctac_nifti.header
        )
        
        # Save the aligned nifti file
        output_path = f"sCT0_LDM36/sCT0_{case_name}_aligned.nii.gz"
        nib.save(aligned_nifti, output_path)
        print(f"Successfully processed {case_name}")
    except FileNotFoundError as e:
        print(f"Error processing {case_name}: {str(e)}")
        continue
    