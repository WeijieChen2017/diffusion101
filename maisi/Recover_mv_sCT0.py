import os
import shutil

# List of cases to process
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
sCT0_src_folder = "../../SharkSeagrass/B100/DLCTAC_bed/"
sCT0_dst_folder = "James_36/sCT0_MLAA"

# Create destination folders if they don't exist
os.makedirs(sCT0_dst_folder, exist_ok=True)

for idx, case_name in enumerate(case_name_list, 1):
    sCT0_src_path = f"{sCT0_src_folder}/{case_name}_CTAC_DL_bed.nii.gz"
    sCT0_dst_path = f"{sCT0_dst_folder}/sCT0_{case_name}_v3_bed.nii.gz"
    
    # Copy file and show progress
    shutil.copy2(sCT0_src_path, sCT0_dst_path)
    print(f"[{idx}/{len(case_name_list)}] Copied {case_name}")
    
