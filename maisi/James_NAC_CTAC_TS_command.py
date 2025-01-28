work_dir = "NAC_CTAC_Spacing15"

case_name_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 'E4242', 'E4275', 'E4298', 'E4313',
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

import os
import nibabel as nib
import numpy as np

# available GPU is 0/1/2/4/5/6
GPU_mapping = ["0", "1", "2", "4", "5", "6"]

for i, case_name in enumerate(case_name_list):
    # NAC_E4055_256.nii.gz CTAC_E4055_256.nii.gz
    CTACIVV_cropped_path = f"{work_dir}/CTAC_{case_name}_cropped.nii.gz"
    CTACIVV_cropped_seg_path = f"{work_dir}/CTAC_{case_name}_TS.nii.gz"
    gpu_idx = f"gpu:{GPU_mapping[i % 4]}"

    command = f"TotalSegmentator -i {CTACIVV_cropped_path} -o {CTACIVV_cropped_seg_path} --device {gpu_idx} --ml"
    print(command)

    



    
