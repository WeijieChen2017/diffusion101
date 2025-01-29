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

region_list = ["whole", "soft", "bone"]

import os
import nibabel as nib
import numpy as np

from scipy.ndimage import binary_fill_holes
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


root_dir = "James_36/synCT/"
mask_dir = f"{root_dir}/mask"
os.makedirs(mask_dir, exist_ok=True)
ct_dir = f"{root_dir}"
con_dir = f"{root_dir}"
synCT_seg_dir = f"{root_dir}"
# save_dir = os.path.join(root_dir, "inference_20250128_noon")
save_dir = f"{root_dir}"
synCT_dir = f"{save_dir}"

min_boundary = -1024
soft_boundary = -450
bone_boundary = 150
max_boundary = 3000

metrics_dict = {
    "mae_by_case": {},
    "mae_by_region": {},
    "ssim_by_case": {},
    "ssim_by_region": {},
    "psnr_by_case": {},
    "psnr_by_region": {},
}

for case_name in case_name_list:

    ct_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"
    body_contour_path = f"{root_dir}/SynCT_{case_name}_TS_body.nii.gz"
    synCT_seg_path = f"{root_dir}/SynCT_{case_name}_TS_label.nii.gz"
    # synCT_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI.nii.gz"
    synCT_path = f"{synCT_dir}/SynCT_{case_name}.nii.gz"

    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    body_contour_file = nib.load(body_contour_path)
    body_contour = body_contour_file.get_fdata()
    synCT_file = nib.load(synCT_path)
    synCT_data = synCT_file.get_fdata()
    
    # compute soft and bone mask
    soft_mask_path = f"{mask_dir}/SynCT_{case_name}_mask_soft.nii.gz"
    bone_mask_path = f"{mask_dir}/SynCT_{case_name}_mask_bone.nii.gz"

    # if exist, load the mask
    if os.path.exists(soft_mask_path):
        soft_mask = nib.load(soft_mask_path).get_fdata()
        bone_mask = nib.load(bone_mask_path).get_fdata()
    else:
        # mask_CT_soft = (CT_GT_data >= HU_boundary_soft[0]) & (CT_GT_data <= HU_boundary_soft[1])
        soft_mask = (synCT_data >= soft_boundary) & (synCT_data <= bone_boundary)
        # mask_CT_bone = (CT_GT_data >= HU_boundary_bone[0]) & (CT_GT_data <= HU_boundary_bone[1])
        bone_mask = (synCT_data >= bone_boundary) & (synCT_data <= max_boundary)
        # save the mask
        soft_mask_nii = nib.Nifti1Image(soft_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(soft_mask_nii, soft_mask_path)
        bone_mask_nii = nib.Nifti1Image(bone_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(bone_mask_nii, bone_mask_path)
        print(f"Saved soft and bone mask for {case_name} at {soft_mask_path} and {bone_mask_path}")

    # compute metrics for whole, soft, and bone regions
    region_list = ["body", "soft", "bone"]
    mask_body_binary = body_contour > 0
    mask_soft_binary = soft_mask > 0
    mask_bone_binary = bone_mask > 0
    mask_list = [mask_body_binary, mask_soft_binary, mask_bone_binary]

    for i in range(len(region_list)):
        region = region_list[i]
        mask = mask_list[i]

        # compute metrics
        mae = np.mean(np.abs(ct_data[mask] - synCT_data[mask]))
        ssim_val = ssim(ct_data, synCT_data, data_range=max_boundary - min_boundary, mask=mask)
        psnr_val = psnr(ct_data, synCT_data, data_range=max_boundary - min_boundary)

        # save metrics
        if case_name not in metrics_dict["mae_by_case"]:
            metrics_dict["mae_by_case"][case_name] = {}
            metrics_dict["ssim_by_case"][case_name] = {}
            metrics_dict["psnr_by_case"][case_name] = {}
        metrics_dict["mae_by_case"][case_name][region] = mae
        metrics_dict["ssim_by_case"][case_name][region] = ssim_val
        metrics_dict["psnr_by_case"][case_name][region] = psnr_val

        if region not in metrics_dict["mae_by_region"]:
            metrics_dict["mae_by_region"][region] = []
            metrics_dict["ssim_by_region"][region] = []
            metrics_dict["psnr_by_region"][region] = []
        metrics_dict["mae_by_region"][region].append(mae)
        metrics_dict["ssim_by_region"][region].append(ssim_val)
        metrics_dict["psnr_by_region"][region].append(psnr_val)

    print(f"Computed metrics for {case_name}, MAE: {mae}, SSIM: {ssim_val}, PSNR: {psnr_val}")

# compute average metrics
for region in region_list:
    metrics_dict["mae_by_region"][region] = np.mean(metrics_dict["mae_by_region"][region])
    metrics_dict["ssim_by_region"][region] = np.mean(metrics_dict["ssim_by_region"][region])
    metrics_dict["psnr_by_region"][region] = np.mean(metrics_dict["psnr_by_region"][region])

print("Average metrics by region:")
print(f"MAE: {metrics_dict['mae_by_region']}")
print(f"SSIM: {metrics_dict['ssim_by_region']}")
print(f"PSNR: {metrics_dict['psnr_by_region']}")
print("Metrics by case:")
print(f"MAE: {metrics_dict['mae_by_case']}")
print(f"SSIM: {metrics_dict['ssim_by_case']}")
print(f"PSNR: {metrics_dict['psnr_by_case']}")

# Save metrics to json
import json
metrics_json_path = f"{root_dir}/metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_dict, f)

print(f"Saved metrics to {metrics_json_path}")
print("Done!")


