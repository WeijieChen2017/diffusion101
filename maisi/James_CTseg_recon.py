# new_CTACIVV_data = CTACIVV_data[21:277, 21:277, :]

import os
import glob
import numpy as np
import nibabel as nib

case_name_list = [
    'E4242', 'E4275', 'E4298', 'E4313',
    'E4245', 'E4276', 'E4299', 'E4317',
    'E4246', 'E4280', 'E4300', 'E4318',
    'E4247', 'E4282', 'E4301', 'E4324',
    'E4248', 'E4283', 'E4302', 'E4325',
    'E4250', 'E4284', 'E4306', 'E4328',
    'E4252', 'E4288', 'E4307', 'E4332',
    'E4259', 'E4289', 'E4308', 'E4335',
    'E4260', 'E4290', 'E4309', 'E4336',
    'E4261', 'E4292', 'E4310', 'E4337',
    'E4273', 'E4297', 'E4312', 'E4338',
]

# Part 0
NAC_folder = "Duetto_Output_B100_part3"
CTAC_folder = "CTAC_IVV"
dst_folder = "NAC_CT_pairs_part3"
os.makedirs(dst_folder, exist_ok=True)

# load every folder in NAC_folder
for case_name in case_name_list:
    print("-"*30)
    # PET TOFNAC E4193 B100 and CTACIVV_4301.nii
    NAC_path = f"{NAC_folder}/PET TOFNAC {case_name} B100/*.nii"
    CTAC_path = f"{CTAC_folder}/CTACIVV_{case_name}.nii"
    # check how many NAC path exist
    NAC_path_list = glob.glob(NAC_path)
    if len(NAC_path_list) == 0:
        print(f"case: {case_name}, NAC path not found!")
    elif len(NAC_path_list) > 1:
        print(f"case: {case_name}, multiple NAC paths found!")
    else:
        # move the NAC and CTAC to the dst folder
        cmd_NAC = f"cp {NAC_path_list[0]} {dst_folder}/NAC_{case_name}.nii"
        cmd_CTAC = f"cp {CTAC_path} {dst_folder}/CTAC_{case_name}.nii"
        print(cmd_NAC)
        print(cmd_CTAC)
        os.system(cmd_NAC)
        os.system(cmd_CTAC)
        print(f"case: {case_name}, NAC and CTAC moved!")
    




# Part 1
# ----> rename all the files in the James_data_v3 directory

# TOFNAC_dir = "James_data_v3/_nii_Winston/"
# CTACIVV_dir = "James_data_v3/part3/"

# TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*/*.nii"))
# for TOFNAC_path in TOFNAC_dir_list:
#     print()
#     case_name = TOFNAC_path.split("/")[-2].split(".")[0]
#     # PET TOFNAC E4237 B100
#     case_name = case_name.split(" ")[-2]
#     CTACIVV_path = f"{CTACIVV_dir}CTACIVV_{case_name[1:]}.nii"
#     print(case_name)
#     move_cmd_TOFNAC = f"mv \"{TOFNAC_path}\" James_data_v3/TOFNAC_{case_name}.nii"
#     move_cmd_CTACIVV = f"mv {CTACIVV_path} James_data_v3/CTACIVV_{case_name}.nii"
#     print(move_cmd_TOFNAC)
#     print(move_cmd_CTACIVV)
#     os.system(move_cmd_TOFNAC)
#     os.system(move_cmd_CTACIVV)

# Part 2
# ----> 3d resample to 2.344 mm isotropic

# TOFNAC_dir = "James_data_v3/TOFNAC/"
# CTACIVV_dir = "James_data_v3/CTACIVV/"

# TOFNAC_dir_list =sorted(glob.glob(TOFNAC_dir + "*.nii"))
# CTACIVV_dir_list =sorted(glob.glob(CTACIVV_dir + "*.nii"))

# for TOFNAC_path in TOFNAC_dir_list:
#     dst_path = TOFNAC_path.replace(".nii", "_256.nii")
#     print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {TOFNAC_path}")

# for CTACIVV_path in CTACIVV_dir_list:
#     dst_path = CTACIVV_path.replace(".nii", "_256.nii")
#     print(f"3dresample -dxyz 2.344 2.344 2.344 -prefix {dst_path} -inset {CTACIVV_path}")

# Part 3
# ----> check the data for dim, min and max

# TOFNAC_dir = "James_data_v3/TOFNAC_256/"
# CTACIVV_dir = "James_data_v3/CTACIVV_256/"

# TOFNAC_path_lists = sorted(glob.glob(TOFNAC_dir + "*.nii"))
# casename_list = []

# for TOFNAC_path in TOFNAC_path_lists:
#     casename = TOFNAC_path.split("/")[-1].split(".")[0]
#     # TOFNAC_E4063_256.nii
#     casename = casename.split("_")[1]
#     casename_list.append(casename)
#     CTACIVV_path = CTACIVV_dir + "CTACIVV_" + casename + "_256.nii"

#     TOFNAC_file = nib.load(TOFNAC_path)
#     CTACIVV_file = nib.load(CTACIVV_path)

#     TOFNAC_data = TOFNAC_file.get_fdata()
#     CTACIVV_data = CTACIVV_file.get_fdata()

#     print(f"case: {casename}, DIM TOFNAC: {TOFNAC_data.shape}, CTACIVV: {CTACIVV_data.shape}")
#     print(f"case: {casename}, MIN TOFNAC: {TOFNAC_data.min()}, CTACIVV: {CTACIVV_data.min()}")
#     print(f"case: {casename}, MAX TOFNAC: {TOFNAC_data.max()}, CTACIVV: {CTACIVV_data.max()}")

#     TOFNAC_file_new = nib.Nifti1Image(TOFNAC_data, TOFNAC_file.affine, TOFNAC_file.header)
#     CTACIVV_file_new = nib.Nifti1Image(CTACIVV_data, CTACIVV_file.affine, CTACIVV_file.header)

#     nib.save(TOFNAC_file_new, TOFNAC_dir + "TOFNAC_" + casename + "_256.nii.gz")
#     nib.save(CTACIVV_file_new, CTACIVV_dir + "CTACIVV_" + casename + "_256.nii.gz")

#     print(f"case: {casename}, saved!")                                                    

# Part 4
# ----> cut CTACIVV data from 299 to 256 in x/y dim, and output casename if z dim is not matched to TOFNAC

# TOFNAC_dir = "James_data_v3/TOFNAC_256/"
# CTACIVV_dir = "James_data_v3/CTACIVV_256/"

# TOFNAC_path_lists = sorted(glob.glob(TOFNAC_dir + "*.nii.gz"))
# casename_list = []

# for TOFNAC_path in TOFNAC_path_lists:
#     casename = TOFNAC_path.split("/")[-1].split(".")[0]
#     # TOFNAC_E4063_256.nii
#     casename = casename.split("_")[1]
#     casename_list.append(casename)
#     CTACIVV_path = CTACIVV_dir + "CTACIVV_" + casename + "_256.nii.gz"

#     TOFNAC_file = nib.load(TOFNAC_path)
#     CTACIVV_file = nib.load(CTACIVV_path)

#     TOFNAC_data = TOFNAC_file.get_fdata()
#     CTACIVV_data = CTACIVV_file.get_fdata()

#     new_CTACIVV_data = CTACIVV_data[21:277, 21:277, :]
#     new_CTACIVV_file = nib.Nifti1Image(new_CTACIVV_data, TOFNAC_file.affine, TOFNAC_file.header)
#     new_CTACIVV_name = CTACIVV_dir + "CTACIVV_" + casename + "_aligned.nii.gz"
#     nib.save(new_CTACIVV_file, new_CTACIVV_name)
#     print(f"case: {new_CTACIVV_name}, saved!")

#     if TOFNAC_data.shape[2] != CTACIVV_data.shape[2]:
#         print(f"case: {casename}, TOFNAC: {TOFNAC_data.shape}, CTACIVV: {CTACIVV_data.shape}")
#         print(f"case: {casename}, z dim not matched!")
