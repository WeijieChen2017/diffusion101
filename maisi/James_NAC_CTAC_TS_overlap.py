case_name_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
    'E4242', 'E4275', 'E4298', 'E4313',
    'E4245', 'E4276', 'E4299', 'E4317', 'E4246',
    'E4280', 'E4300', 'E4318', 'E4247', 'E4282',
    'E4301', 'E4324', 'E4248', 'E4283', 'E4302',
    'E4325', 'E4250', 'E4284', 'E4306', 'E4328',
    'E4252', 'E4288', 'E4307', 'E4332', 'E4259',
    'E4308', 'E4335', 'E4260', 'E4290', 'E4309',
    'E4336', 'E4261', 'E4292', 'E4310', 'E4337',
    'E4273', 'E4297', 'E4312', 'E4338',
]
# E4063, E4080, E4087, E4097, E4102, E4289 are removed for z mismatch

case_name_list = sorted(case_name_list)

# Mapping dictionary (unchanged)
T2M_mapping = {
    1: 3, 2: 5, 3: 14, 4: 10, 5: 1, 6: 12,
    7: 4, 8: 8, 9: 9, 10: 28, 11: 29, 12: 30,
    13: 31, 14: 32, 15: 11, 16: 57, 17: 126, 18: 19,
    19: 13, 20: 62, 21: 15, 22: 118, 23: 116, 24: 117,
    25: 97, 26: 127, 27: 33, 28: 34, 29: 35, 30: 36,
    31: 37, 32: 38, 33: 39, 34: 40, 35: 41, 36: 42,
    37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
    43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 
    49: 55, 50: 56, 51: 115, 52: 6, 53: 119, 54: 109, 
    55: 123, 56: 124, 57: 112, 58: 113, 59: 110, 60: 111, 
    61: 108, 62: 125, 63: 7, 64: 17, 65: 58, 66: 59, 
    67: 60, 68: 61, 69: 87, 70: 88, 71: 89, 72: 90, 
    73: 91, 74: 92, 75: 93, 76: 94, 77: 95, 78: 96, 
    79: 121, 80: 98, 81: 99, 82: 100, 83: 101, 84: 102, 
    85: 103, 86: 104, 87: 105, 88: 106, 89: 107, 90: 22, 
    91: 120, 92: 63, 93: 64, 94: 65, 95: 66, 96: 67, 
    97: 68, 98: 69, 99: 70, 100: 71, 101: 72, 102: 73, 
    103: 74, 104: 75, 105: 76, 106: 77, 107: 78, 108: 79, 
    109: 80, 110: 81, 111: 82, 112: 83, 113: 84, 114: 85, 
    115: 86, 116: 122, 117: 114,
}

import os
import nibabel as nib
import numpy as np

from scipy.ndimage import binary_fill_holes


# available GPU is 0/1/2/4/5/6
GPU_mapping = ["0", "1", "2", "4", "5", "6"]
commands_list = [
    [],
    [],
    [],
    [],
]
work_dir = "NAC_CTAC_Spacing15"

# available GPU is 5/6
# GPU_mapping = ["5", "6"]
# commands_list = [
#     [],
#     [],
#     [],
#     [],
# ]
# work_dir = "James_36/synCT"
for i, case_name in enumerate(case_name_list):
    # NAC_E4055_256.nii.gz CTAC_E4055_256.nii.gz
    # CTACIVV_cropped_path = f"{work_dir}/CTAC_{case_name}_cropped.nii.gz"
    # CTACIVV_cropped_seg_path = f"{work_dir}/CTAC_{case_name}_TS.nii.gz"
    CTACIVV_cropped_path = f"{work_dir}/SynCT_{case_name}.nii.gz"
    CTACIVV_cropped_seg_path = f"{work_dir}/SynCT_{case_name}_TS.nii.gz"
    # gpu_idx = f"gpu:{GPU_mapping[i % 2]}"

    # command = f"TotalSegmentator -i {CTACIVV_cropped_path} -o {CTACIVV_cropped_seg_path} --device {gpu_idx} --ml"
    # commands_list[i % 2].append(command)

# for i, commands in enumerate(commands_list):
#     print(f"echo 'Running on GPU {GPU_mapping[i]}'")
#     for command in commands:
#         print(command)
#     print("\n")

    CT_file = nib.load(CTACIVV_cropped_path)
    CT_data = CT_file.get_fdata()
    seg_file = nib.load(CTACIVV_cropped_seg_path)
    seg_data = seg_file.get_fdata()

    # get body contour
    body_contour = np.zeros_like(seg_data, dtype=np.uint16)
    len_z = body_contour.shape[2]
    for z in range(len_z):
        slice_z = body_contour[:, :, z]
        slice_z[CT_data[:, :, z] > -500] = 1
        slice_z = binary_fill_holes(slice_z)
        body_contour[:, :, z] = slice_z

    body_contour_nii = nib.Nifti1Image(body_contour, seg_file.affine, seg_file.header)
    body_contour_path = CTACIVV_cropped_seg_path.replace("_TS.nii.gz", "_TS_body.nii.gz")
    nib.save(body_contour_nii, body_contour_path)
    print(f"Saved body contour to {body_contour_path}")

    # copy as label overlap
    label_overlap = body_contour.copy()
    label_overlap[body_contour > 0] = 200
    
    # label transformation using T2M_mapping
    n_labels = len(T2M_mapping.keys())
    for i in range(1, n_labels + 1):
        label_overlap[seg_data == i] = T2M_mapping[i]
    
    label_overlap_nii = nib.Nifti1Image(label_overlap, seg_file.affine, seg_file.header)
    label_overlap_path = CTACIVV_cropped_seg_path.replace("_TS.nii.gz", "_TS_label.nii.gz")
    nib.save(label_overlap_nii, label_overlap_path)
    print(f"Saved label overlap to {label_overlap_path}")
    



    


    



    
