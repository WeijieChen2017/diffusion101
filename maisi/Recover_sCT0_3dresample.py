# sCT0_E4058_aligned

import glob
import os
import nibabel as nib

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

sCT0_dst_folder = "James_36/sCT0_MLAA"

# Create destination folders if they don't exist
os.makedirs(sCT0_dst_folder, exist_ok=True)

CT_bed_folder = "CTAC_bed/"

for case_name in case_name_list:
    sCT1_src_path = f"sCT0_LDM36/sCT0_{case_name}_aligned.nii.gz"

    sCT1_dst_path = f"{sCT0_dst_folder}/sCT0_{case_name}_v1_3dresample.nii.gz"

    CT_bed_path = sorted(glob.glob(f"{CT_bed_folder}*_{case_name[1:]}_*.nii"))[0]
    CT_bed_nifti = nib.load(CT_bed_path)
    
    # Get pixel dimensions from the header
    dx, dy, dz = CT_bed_nifti.header['pixdim'][1:4]
    
    # Format dimensions to 4 decimal places
    dx = f"{dx:.4f}"
    dy = f"{dy:.4f}"
    dz = f"{dz:.4f}"
    
    # 3dresample commands for both sCT1 and sCT2
    command1 = f"3dresample -dxyz {dx} {dy} {dz} -prefix {sCT1_dst_path} -input {sCT1_src_path}"
    
    # output commands
    print(command1)
