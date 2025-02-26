import os
import shutil
import nibabel as nib
import numpy as np

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
CTAC_maisi_src_folder = "James_36/CTAC_maisi"
sCT0_src_folder = "sCT0_LDM36"
sCT1_src_folder = "James_36/sCT1_MLAA"
sCT2_src_folder = "James_36/sCT2_MLAA"
output_folder = "James_recon_no_bed"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print(f"Starting processing {len(case_name_list)} cases")

for idx, case_name in enumerate(case_name_list, 1):
    print(f"[{idx}/{len(case_name_list)}] Processing case: {case_name}")

    # Define source and destination paths
    CTAC_maisi_src_path = f"{CTAC_maisi_src_folder}/CTAC_maisi_{case_name}_v1_3dresample.nii.gz"
    sCT0_src_path = f"{sCT0_src_folder}/sCT0_{case_name}_400.nii.gz"
    sCT1_src_path = f"{sCT1_src_folder}/sCT1_{case_name}_v1_3dresample.nii.gz"
    sCT2_src_path = f"{sCT2_src_folder}/sCT2_{case_name}_v1_3dresample.nii.gz"
    
    CTAC_maisi_dst_path = f"{output_folder}/CTAC_maisi_{case_name}_no_bed.nii.gz"
    sCT0_dst_path = f"{output_folder}/sCT0_{case_name}_no_bed.nii.gz"
    sCT1_dst_path = f"{output_folder}/sCT1_{case_name}_no_bed.nii.gz"
    sCT2_dst_path = f"{output_folder}/sCT2_{case_name}_no_bed.nii.gz"
    
    try:
        # Process CTAC_maisi
        if os.path.exists(CTAC_maisi_src_path):
            print(f"  Copying CTAC_maisi for {case_name}...")
            shutil.copy(CTAC_maisi_src_path, CTAC_maisi_dst_path)
            print(f"  Saved: {CTAC_maisi_dst_path}")
        else:
            print(f"  CTAC_maisi file not found for {case_name}")

        # Process sCT0
        if os.path.exists(sCT0_src_path):
            print(f"  Copying sCT0 for {case_name}...")
            shutil.copy(sCT0_src_path, sCT0_dst_path)
            print(f"  Saved: {sCT0_dst_path}")
        else:
            print(f"  sCT0 file not found for {case_name}")

        # Process sCT1
        if os.path.exists(sCT1_src_path):
            print(f"  Copying sCT1 for {case_name}...")
            shutil.copy(sCT1_src_path, sCT1_dst_path)
            print(f"  Saved: {sCT1_dst_path}")
        else:
            print(f"  sCT1 file not found for {case_name}")

        # Process sCT2
        if os.path.exists(sCT2_src_path):
            print(f"  Copying sCT2 for {case_name}...")
            shutil.copy(sCT2_src_path, sCT2_dst_path)
            print(f"  Saved: {sCT2_dst_path}")
        else:
            print(f"  sCT2 file not found for {case_name}")
        
    except Exception as e:
        print(f"  Error processing case {case_name}: {str(e)}")
        continue
    
    print(f"Completed case: {case_name}\n")

print("Processing complete!") 