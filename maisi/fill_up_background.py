# seg_map_folder = "CTACIVV_MAISeg" # 512, 512, 335 1.367, 1.367, 3.27
# body_contour_folder = "../James_data_v3/mask"
# find every .nii.gz file in this folder

# load all files in the target folder
import os
import glob
import nibabel as nib

# file_list = sorted(glob.glob(body_contour_folder+"/mask_body_contour_*.nii.gz"))
# print(f"We find {len(file_list)} files in the target folder")
# for file in file_list:
#     new_name = file.replace(".nii.gz", "_Spacing15.nii.gz")
#     command = f"3dresample -dxyz 1.5 1.5 1.5 -rmode NN -prefix {new_name} -input {file}"
#     print(command)

folder = "Spacing15"
seg_file_list = sorted(glob.glob(folder+"/CTACIVV_*_MAISI_Spacing15.nii.gz"))
for seg_path in seg_file_list:
    case_name = os.path.basename(seg_path)[8:13]
    print(case_name)
    body_contour_path = folder+f"/mask_body_contour_{case_name}_Spacing15.nii.gz"
    print(body_contour_path)
    seg_file = nib.load(seg_path)
    con_file = nib.load(body_contour_path)
    seg_data = seg_file.fdata()
    con_data = con_file.fdata()
    print(f"seg shape: {seg_data.shape}, con shape: {con_data.shape}")