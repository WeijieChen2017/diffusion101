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

# Define input/output directories
CTAC_maisi_folder = "NAC_CTAC_Spacing15/inference_20250128_noon"
CTAC_maisi_dst_folder = "James_36/CTAC_maisi"

# Create destination folders if they don't exist
os.makedirs(CTAC_maisi_dst_folder, exist_ok=True)

# Define CT_bed folder
CT_bed_folder = "CTAC_bed/"

for case_name in case_name_list:

    # load CTAC_maisi
    CTAC_maisi_src_path = f"{CTAC_maisi_folder}/CTAC_{case_name}_TS_MAISI.nii.gz"
    CTAC_maisi_dst_path = f"{CTAC_maisi_dst_folder}/CTAC_maisi_{case_name}_v1_3dresample.nii.gz"

    # load CT_bed
    CT_bed_path = sorted(glob.glob(f"{CT_bed_folder}*_{case_name[1:]}_*.nii"))[0]
    CT_bed_nifti = nib.load(CT_bed_path)
    
    # Get pixel dimensions from the header
    dx, dy, dz = CT_bed_nifti.header['pixdim'][1:4]
    
    # Format dimensions to 4 decimal places
    dx = f"{dx:.4f}"
    dy = f"{dy:.4f}"
    dz = f"{dz:.4f}"
    
    # 3dresample commands for both CTAC_maisi
    command1 = f"3dresample -dxyz {dx} {dy} {dz} -prefix {CTAC_maisi_dst_path} -input {CTAC_maisi_src_path}"
    
    # output commands
    print(command1)
