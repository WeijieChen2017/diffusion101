case_list_total = [
    "BII096","BPO124","DZS099",
    "EGS066","EIA058","FGX078",
    "FNG137","GSB106","HNJ120",
    "JLB061","JQR130","KQA094",
    "KWX131","KZF084","LBO118",
    "LCQ128","MLU077","NAF069",
    "NIR103","NKQ091","ONC134",
    "OOP125","PAW055","RFK139",
    "RSE114","SAM079","SCH068",
    "SNF129","SPT074","TTE081",
    "TVA105","WLX138","WVX115",
    "XZG098","YKY073","ZTS092",
]

work_dir = "../ISBI_test/sCT_A"
dst_dir = "sCT0_LDM36"

# for each casename, find the {casename}*_corrected.nii.gz

import glob
import os

os.makedirs(dst_dir, exist_ok=True)

for case_name in case_list_total:
    file_path = glob.glob(f"{work_dir}/{case_name}*_corrected.nii.gz")[0]
    print(file_path)
    new_name = f"sCT0_E4{case_name[3:]}.nii.gz"
    cmd = f"cp {file_path} {dst_dir}/{new_name}"
    print(cmd)
    os.system(cmd)
