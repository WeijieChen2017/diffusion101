# nii2dcm nifti-file.nii.gz dicom-output-directory/ --dicom-type CT -r reference-dicom-file.dcm
# wxc321@l-mimrtl02539:/shares/mimrtl/Users/James/'synthetic CT PET AC'$

# case_name_list = [
#     'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
#     'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
#     'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
#     'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
#     'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
#     'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
#     'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
#     'E4139', 
# ]
case_name_list = [
    'E4058',
]

import os
import glob

nii_folder = "sCT_NAC"

for case_name in case_name_list:
    nii_file = sorted(glob.glob(f'Test_1sub_sCTnii/{nii_folder}/*{case_name}*.nii.gz'))[0]
    dicom_output_directory = f'Test_1sub_sCTdcm/{nii_folder}/PETLYMPH_{case_name[1:]}'
    reference_dicom_file = f'Duetto_Output_B100/PETLYMPH_{case_name[1:]}'
    # command = f'nii2dcm {nii_file} {dicom_output_directory} --dicom-type CT -r {reference_dicom_file}'

    print(f"[{case_name_list.index(case_name)+1}/{len(case_name_list)}]")
    print(f"nii_file: {nii_file}")
    print(f"dicom_output_directory: {dicom_output_directory}")
    print(f"reference_dicom_file: {reference_dicom_file}")
