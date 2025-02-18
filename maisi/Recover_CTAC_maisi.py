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

# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4055_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4055_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4058_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4058_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4061_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4061_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4066_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4066_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4068_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4068_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4069_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4069_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4073_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4073_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4074_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4074_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4077_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4077_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4078_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4078_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4079_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4079_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4081_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4081_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4084_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4084_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4091_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4091_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4092_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4092_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4094_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4094_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4096_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4096_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4098_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4098_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4099_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4099_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4103_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4103_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4105_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4105_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4106_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4106_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4114_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4114_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4115_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4115_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4118_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4118_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4120_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4120_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4124_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4124_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4125_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4125_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4128_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4128_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4129_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4129_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4130_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4130_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4131_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4131_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4134_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4134_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4137_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4137_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4138_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4138_TS_MAISI.nii.gz
# 3dresample -dxyz 1.3672 1.3672 3.2700 -prefix James_36/CTAC_maisi/CTAC_maisi_E4139_v1_3dresample.nii.gz -input NAC_CTAC_Spacing15/inference_20250128_noon/CTAC_E4139_TS_MAISI.nii.gz