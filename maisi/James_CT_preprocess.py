work_dir = "NAC_CTAC_Spacing15"

case_name_list = [
    "E4055", "E4058", "E4061", "E4063", "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079", "E4080", "E4081", "E4084",
    "E4087", "E4091", "E4092", "E4094", "E4096",
    "E4097", "E4098", "E4099", "E4102", "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
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

import os
import nibabel as nib
import numpy as np
import pandas as pd
import xlsxwriter

xlsx_path = f"{work_dir}/NAC_CTAC_Spacing15.xlsx"
workbook = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')

# write down the header
header = [
    "case_name",
    "TOFNAC_shape_x", "TOFNAC_shape_y", "TOFNAC_shape_z",
    "CTACIVV_shape_x", "CTACIVV_shape_y", "CTACIVV_shape_z",
    "TOFNAC_spacing_x", "TOFNAC_spacing_y", "TOFNAC_spacing_z",
    "CTACIVV_spacing_x", "CTACIVV_spacing_y", "CTACIVV_spacing_z",
    "TOFNAC_physical_space_x", "TOFNAC_physical_space_y", "TOFNAC_physical_space_z",
    "CTACIVV_physical_space_x", "CTACIVV_physical_space_y", "CTACIVV_physical_space_z",
    "TOFNAC_min", "TOFNAC_max", "TOFNAC_mean", "TOFNAC_std",
    "CTACIVV_min", "CTACIVV_max", "CTACIVV_mean", "CTACIVV_std",
]

for i, case_name in enumerate(case_name_list):
    # NAC_E4055_256.nii.gz CTAC_E4055_256.nii.gz
    TOFNAC_path = f"{work_dir}/NAC_{case_name}_256.nii.gz"
    CTACIVV_path = f"{work_dir}/CTAC_{case_name}_256.nii.gz"

    TOFNAC_file = nib.load(TOFNAC_path)
    TOFNAC_data = TOFNAC_file.get_fdata()
    CTACIVV_file = nib.load(CTACIVV_path)
    CTACIVV_data = CTACIVV_file.get_fdata()

    # # find the spacing of the two images
    # TOFNAC_spacing = TOFNAC_file.header.get_zooms()
    # CTACIVV_spacing = CTACIVV_file.header.get_zooms()

    # # compute the physical space of the two images
    # TOFNAC_physical_space = np.array(TOFNAC_spacing) * np.array(TOFNAC_data.shape)
    # CTACIVV_physical_space = np.array(CTACIVV_spacing) * np.array(CTACIVV_data.shape)

    # write down xlsx of shape, spacing, physical space, min, max, mean, std
    TOFNAC_shape = TOFNAC_data.shape
    CTACIVV_shape = CTACIVV_data.shape
    TOFNAC_spacing = TOFNAC_file.header.get_zooms()
    CTACIVV_spacing = CTACIVV_file.header.get_zooms()
    TOFNAC_physical_space = np.array(TOFNAC_spacing) * np.array(TOFNAC_data.shape)
    CTACIVV_physical_space = np.array(CTACIVV_spacing) * np.array(CTACIVV_data.shape)

    TOFNAC_min = np.min(TOFNAC_data)
    TOFNAC_max = np.max(TOFNAC_data)
    TOFNAC_mean = np.mean(TOFNAC_data)
    TOFNAC_std = np.std(TOFNAC_data)
    
    CTACIVV_min = np.min(CTACIVV_data)
    CTACIVV_max = np.max(CTACIVV_data)
    CTACIVV_mean = np.mean(CTACIVV_data)
    CTACIVV_std = np.std(CTACIVV_data)

    data = [
        case_name,
        TOFNAC_shape[0], TOFNAC_shape[1], TOFNAC_shape[2],
        CTACIVV_shape[0], CTACIVV_shape[1], CTACIVV_shape[2],
        TOFNAC_spacing[0], TOFNAC_spacing[1], TOFNAC_spacing[2],
        CTACIVV_spacing[0], CTACIVV_spacing[1], CTACIVV_spacing[2],
        TOFNAC_physical_space[0], TOFNAC_physical_space[1], TOFNAC_physical_space[2],
        CTACIVV_physical_space[0], CTACIVV_physical_space[1], CTACIVV_physical_space[2],
        TOFNAC_min, TOFNAC_max, TOFNAC_mean, TOFNAC_std,
        CTACIVV_min, CTACIVV_max, CTACIVV_mean, CTACIVV_std,
    ]

    df = pd.DataFrame([data], columns=header)
    df.to_excel(workbook, sheet_name=case_name, index=False)
    print(f"case: {case_name}, processed!")
    # print(f"case: {case_name}, TOFNAC shape: {TOFNAC_data.shape}, CTACIVV shape: {CTACIVV_data.shape}")
    # print(f"case: {case_name}, TOFNAC spacing: {TOFNAC_spacing}, CTACIVV spacing: {CTACIVV_spacing}")
    # print(f"case: {case_name}, TOFNAC physical space: {TOFNAC_physical_space}, CTACIVV physical space: {CTACIVV_physical_space}")
    # print("-" * 50)

workbook.save()
workbook.close()
print(f"Saved to {xlsx_path}")



    
