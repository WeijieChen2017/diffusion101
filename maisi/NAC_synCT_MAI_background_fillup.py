work_dir = "NAC_synCT_MAISI"

# BII096_seg_MAISI_Spacing15.nii.gz  KWX131_seg_MAISI_Spacing15.nii.gz  RSE114_seg_MAISI_Spacing15.nii.gz
# BPO124_seg_MAISI_Spacing15.nii.gz  KZF084_seg_MAISI_Spacing15.nii.gz  SAM079_seg_MAISI_Spacing15.nii.gz
# DZS099_seg_MAISI_Spacing15.nii.gz  LBO118_seg_MAISI_Spacing15.nii.gz  SCH068_seg_MAISI_Spacing15.nii.gz
# EGS066_seg_MAISI_Spacing15.nii.gz  LCQ128_seg_MAISI_Spacing15.nii.gz  SNF129_seg_MAISI_Spacing15.nii.gz
# EIA058_seg_MAISI_Spacing15.nii.gz  MLU077_seg_MAISI_Spacing15.nii.gz  SPT074_seg_MAISI_Spacing15.nii.gz
# FGX078_seg_MAISI_Spacing15.nii.gz  NAF069_seg_MAISI_Spacing15.nii.gz  TTE081_seg_MAISI_Spacing15.nii.gz
# FNG137_seg_MAISI_Spacing15.nii.gz  NIR103_seg_MAISI_Spacing15.nii.gz  TVA105_seg_MAISI_Spacing15.nii.gz
# GSB106_seg_MAISI_Spacing15.nii.gz  NKQ091_seg_MAISI_Spacing15.nii.gz  WLX138_seg_MAISI_Spacing15.nii.gz
# HNJ120_seg_MAISI_Spacing15.nii.gz  ONC134_seg_MAISI_Spacing15.nii.gz  WVX115_seg_MAISI_Spacing15.nii.gz
# JLB061_seg_MAISI_Spacing15.nii.gz  OOP125_seg_MAISI_Spacing15.nii.gz  XZG098_seg_MAISI_Spacing15.nii.gz
# JQR130_seg_MAISI_Spacing15.nii.gz  PAW055_seg_MAISI_Spacing15.nii.gz  YKY073_seg_MAISI_Spacing15.nii.gz
# KQA094_seg_MAISI_Spacing15.nii.gz  RFK139_seg_MAISI_Spacing15.nii.gz  ZTS092_seg_MAISI_Spacing15.nii.gz

case_name_list = [
    "BII096",
    "BPO124",
    "DZS099", 
    "EGS066",
    "EIA058",
    "FGX078",
    "FNG137",
    "GSB106",
    "HNJ120",
    "JLB061",
    "JQR130",
    "KQA094",
    "KWX131",
    "KZF084",
    "LBO118",
    "LCQ128",
    "MLU077",
    "NAF069",
    "NIR103",
    "NKQ091",
    "ONC134",
    "OOP125",
    "PAW055",
    "RFK139",
    "RSE114",
    "SAM079",
    "SCH068",
    "SNF129",
    "SPT074",
    "TTE081",
    "TVA105",
    "WLX138",
    "WVX115",
    "XZG098",
    "YKY073",
    "ZTS092",
]

case_idx_list = [ f"E4{x[3:]}" for x in case_name_list ]

# synCT_seg_list = [ f"{work_dir}/{case_name}_seg_MAISI_Spacing15.nii.gz" for case_name in case_name_list ]
# mask_body_contour_E4241_Spacing15.nii.gz
# body_contour_list = [ f"{work_dir}/mask_body_contour_{case_name}_Spacing15.nii.gz" for case_name in case_name_list ]

import os
import nibabel as nib
import numpy as np

vanila_overlap_save_folder = f"{work_dir}/vanila_overlap"
PET_observed_range_save_folder = f"{work_dir}/PET_observation"

os.makedirs(vanila_overlap_save_folder, exist_ok=True)
os.makedirs(PET_observed_range_save_folder, exist_ok=True)

for case_name in case_name_list:
    case_idx = f"E4{case_name[3:]}"
    synCT_seg_path = f"{work_dir}/{case_name}_seg_MAISI_Spacing15.nii.gz"
    body_contour_path = f"{work_dir}/mask_body_contour_{case_idx}_Spacing15.nii.gz"

    synCT_seg_file = nib.load(synCT_seg_path)
    body_contour_file = nib.load(body_contour_path)

    synCT_seg_data = synCT_seg_file.get_fdata()
    body_contour_data = body_contour_file.get_fdata()

    print(f"Original synCT_seg shape: {synCT_seg_data.shape}, body_contour shape: {body_contour_data.shape}")

    z_len = body_contour_data.shape[2]
    synCT_seg_data_aligned = synCT_seg_data[:, :, :z_len]

    # 1, overlap synCT_seg_data with body_contour_data
    vanila_overlap = np.zeros_like(synCT_seg_data_aligned)
    vanila_overlap[body_contour_data > 0.5] = 200
    vanila_overlap[synCT_seg_data_aligned > 0] = synCT_seg_data_aligned[synCT_seg_data_aligned > 0]
    # use body_contour_file header and affine to save the vanila_overlap
    vanila_overlap_nifti = nib.Nifti1Image(vanila_overlap, body_contour_file.affine, body_contour_file.header)
    vanila_overlap_path = f"{vanila_overlap_save_folder}/vanila_overlap_{case_idx}_Spacing15.nii.gz"
    nib.save(vanila_overlap_nifti, vanila_overlap_path)
    print(f"Saved vanila_overlap to {vanila_overlap_path}")

    # 2, overlap synCT_seg_data in the PET_observed_range with body_contour_data
    PET_observed_range = [
        115, #heart
        23, 28, 29, 30, 31, 32, #lung
        22, 120, #brain and skull
        1, #liver
        3, #spleen
        15, #bladder
        12, 13, 19, #stomach, duodenum, small bowel,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, # spine
    ]
    PET_observed_range_overlap = np.zeros_like(synCT_seg_data_aligned)
    PET_observed_range_overlap[body_contour_data > 0.5] = 200
    # only overlap the PET_observed_range
    for label in PET_observed_range:
        PET_observed_range_overlap[synCT_seg_data_aligned == label] = label
    
    # use body_contour_file header and affine to save the PET_observed_range_overlap
    PET_observed_range_overlap_nifti = nib.Nifti1Image(PET_observed_range_overlap, body_contour_file.affine, body_contour_file.header)
    PET_observed_range_overlap_path = f"{PET_observed_range_save_folder}/PET_observed_range_{case_idx}_Spacing15.nii.gz"
    nib.save(PET_observed_range_overlap_nifti, PET_observed_range_overlap_path)
    print(f"Saved PET_observed_range_overlap to {PET_observed_range_overlap_path}")




