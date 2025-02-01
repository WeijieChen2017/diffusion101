# case_list_total = [
#     "BII096","BPO124","DZS099",
#     "EGS066","EIA058","FGX078",
#     "FNG137","GSB106","HNJ120",
#     "JLB061","JQR130","KQA094",
#     "KWX131","KZF084","LBO118",
#     "LCQ128","MLU077","NAF069",
#     "NIR103","NKQ091","ONC134",
#     "OOP125","PAW055","RFK139",
#     "RSE114","SAM079","SCH068",
#     "SNF129","SPT074","TTE081",
#     "TVA105","WLX138","WVX115",
#     "XZG098","YKY073","ZTS092",
# ]

# work_dir = "../ISBI_test/sCT_A"
dst_dir = "sCT0_LDM36"

# for each casename, find the {casename}*_corrected.nii.gz

import glob
import os

# os.makedirs(dst_dir, exist_ok=True)

# for case_name in case_list_total:
#     file_path = glob.glob(f"{work_dir}/{case_name}*_corrected.nii.gz")[0]
#     print(file_path)
#     new_name = f"sCT0_E4{case_name[3:]}.nii.gz"
#     cmd = f"cp {file_path} {dst_dir}/{new_name}"
#     print(cmd)
#     os.system(cmd)

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

for casename in case_name_list:
    
    path_src = f"{dst_dir}/sCT0_{casename}.nii.gz"
    path_dst = f"{dst_dir}/sCT0_{casename}_400.nii.gz"
    cmd_3dresample = f"3dresample -dxyz 1.5 1.5 1.5 -prefix {path_dst} -inset {path_src}"
    print(cmd_3dresample)