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
from scipy.ndimage import sobel

from scipy.ndimage import binary_fill_holes
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


root_dir = "NAC_CTAC_Spacing15"
mask_dir = f"NAC_CTAC_Spacing15/CT_mask"
os.makedirs(mask_dir, exist_ok=True)
ct_dir = f"{root_dir}"
con_dir = f"{root_dir}"
synCT_seg_dir = f"{root_dir}/inference_20250128_noon"
# save_dir = os.path.join(root_dir, "inference_20250128_noon")
save_dir = f"{root_dir}"
synCT_dir = f"{synCT_seg_dir}"

body_contour_boundary = -500
min_boundary = -1024
soft_boundary = -500
bone_boundary = 300
max_boundary = 3000

metrics_dict = {
    "mae_by_case": {},
    "mae_by_region": {},
    "ssim_by_case": {},
    "ssim_by_region": {},
    "psnr_by_case": {},
    "psnr_by_region": {},
    "dsc_by_case": {},
    "dsc_by_region": {},
    " acutance_by_case": {},
    " acutance_by_region": {},
}

HU_value_adjustment_path = "sCT_CT_stats.npy"
HU_value_adjustment = np.load(HU_value_adjustment_path, allow_pickle=True).item()
ct_mask_overwrite = False
sct_mask_overwrite = False
HU_adjustment = False

for case_name in case_name_list:

    ct_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"
    # synCT_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI.nii.gz"
    synCT_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI.nii.gz"
    synCT_seg_path = f"{root_dir}/CTAC_{case_name}_TS_label.nii.gz"

    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    synCT_file = nib.load(synCT_path)
    synCT_data = synCT_file.get_fdata()
    if HU_adjustment:
        synCT_seg_file = nib.load(synCT_seg_path)
        synCT_seg_data = synCT_seg_file.get_fdata()

    # pad synCT_data according to ct_data, or crop synCT_data to ct_data according
    if ct_data.shape[2] > synCT_data.shape[2]:
        synCT_data = np.pad(synCT_data, ((0, 0), (0, 0), (0, ct_data.shape[2] - synCT_data.shape[2])), mode="constant", constant_values=-1024)
    else:
        synCT_data = synCT_data[:, :, :ct_data.shape[2]]
    
    # compute soft and bone masks from gt CT
    body_mask_path = f"{mask_dir}/mask_body_contour_{case_name}.nii.gz"
    soft_mask_path = f"{mask_dir}/mask_body_soft_{case_name}.nii.gz"
    bone_mask_path = f"{mask_dir}/mask_body_bone_{case_name}.nii.gz"

    # if exist, load the mask
    if os.path.exists(soft_mask_path) and not ct_mask_overwrite:
        body_mask = nib.load(body_mask_path).get_fdata()
        soft_mask = nib.load(soft_mask_path).get_fdata()
        bone_mask = nib.load(bone_mask_path).get_fdata()
    else:
        body_mask = ct_data >= body_contour_boundary
        for i in range(body_mask.shape[2]):
            body_mask[:, :, i] = binary_fill_holes(body_mask[:, :, i])
        # mask_CT_soft = (CT_GT_data >= HU_boundary_soft[0]) & (CT_GT_data <= HU_boundary_soft[1])
        soft_mask = (ct_data >= soft_boundary) & (ct_data <= bone_boundary)
        # mask_CT_bone = (CT_GT_data >= HU_boundary_bone[0]) & (CT_GT_data <= HU_boundary_bone[1])
        bone_mask = (ct_data >= bone_boundary) & (ct_data <= max_boundary)
        # save the mask
        body_mask_nii = nib.Nifti1Image(body_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(body_mask_nii, body_mask_path)
        soft_mask_nii = nib.Nifti1Image(soft_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(soft_mask_nii, soft_mask_path)
        bone_mask_nii = nib.Nifti1Image(bone_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(bone_mask_nii, bone_mask_path)
        # print(f"Saved soft and bone mask for {case_name} at {soft_mask_path} and {bone_mask_path}")

    # HU value adjustment
    if HU_adjustment:
        for key in HU_value_adjustment.keys():
            class_synCT_mean = HU_value_adjustment[key]["sCT_mean"]
            class_synCT_std = HU_value_adjustment[key]["sCT_std"]
            class_CT_mean = HU_value_adjustment[key]["CT_mean"]
            class_CT_std = HU_value_adjustment[key]["CT_std"]
            class_mask = synCT_seg_data == key

            # adjust the HU value of synCT_data
            synCT_data[class_mask] = (synCT_data[class_mask] - class_synCT_mean) * class_CT_std / class_synCT_std + class_CT_mean
        adjusted_nii = nib.Nifti1Image(synCT_data, synCT_file.affine, synCT_file.header)
        adjusted_path = synCT_path.replace(".nii.gz", "_adjusted.nii.gz")
        nib.save(adjusted_nii, adjusted_path)

    # compute metrics for whole, soft, and bone regions
    region_list = ["body", "soft", "bone"]
    mask_body_binary = body_mask > 0
    mask_soft_binary = soft_mask > 0
    mask_bone_binary = bone_mask > 0
    mask_list = [mask_body_binary, mask_soft_binary, mask_bone_binary]

    # compute mask for each region for synCT
    pred_mask = []
    if HU_adjustment:
        pred_body_countour_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_body_adjusted.nii.gz"
        pred_soft_mask_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_mask_soft_adjusted.nii.gz"
        pred_bone_mask_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_mask_bone_adjusted.nii.gz"
    else:
        pred_body_countour_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_body.nii.gz"
        pred_soft_mask_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_mask_soft.nii.gz"
        pred_bone_mask_path = f"{synCT_seg_dir}/SynCT_{case_name}_TS_mask_bone.nii.gz"

    if os.path.exists(pred_body_countour_path) and not sct_mask_overwrite:
        pred_body_countour_file = nib.load(pred_body_countour_path)
        pred_body_countour_data = pred_body_countour_file.get_fdata()
    else:
        pred_body_countour_data = synCT_data >= body_contour_boundary
        for i in range(pred_body_countour_data.shape[2]):
            pred_body_countour_data[:, :, i] = binary_fill_holes(pred_body_countour_data[:, :, i])
        pred_body_countour_nii = nib.Nifti1Image(pred_body_countour_data.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(pred_body_countour_nii, pred_body_countour_path)

    # pad the mask according to body_mask
    if ct_data.shape[2] > pred_body_countour_data.shape[2]:
        pred_body_countour_data = np.pad(pred_body_countour_data, ((0, 0), (0, 0), (0, ct_data.shape[2] - pred_body_countour_data.shape[2])), mode="constant", constant_values=0)
    else:
        pred_body_countour_data = pred_body_countour_data[:, :, :ct_data.shape[2]]
    pred_body_countour_binary = pred_body_countour_data > 0

    if os.path.exists(pred_soft_mask_path) and not sct_mask_overwrite:
        pred_soft_mask = nib.load(pred_soft_mask_path).get_fdata()
        pred_bone_mask = nib.load(pred_bone_mask_path).get_fdata()
    else:
        pred_soft_mask = (synCT_data >= soft_boundary) & (synCT_data <= bone_boundary)
        pred_bone_mask = (synCT_data >= bone_boundary) & (synCT_data <= max_boundary)
        pred_soft_mask_nii = nib.Nifti1Image(pred_soft_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(pred_soft_mask_nii, pred_soft_mask_path)
        pred_bone_mask_nii = nib.Nifti1Image(pred_bone_mask.astype(np.uint8), ct_file.affine, ct_file.header)
        nib.save(pred_bone_mask_nii, pred_bone_mask_path)
        # print(f"Saved soft and bone mask for {case_name} at {pred_soft_mask_path} and {pred_bone_mask_path}")
    
    pred_soft_mask_binary = pred_soft_mask > 0
    pred_bone_mask_binary = pred_bone_mask > 0
    pred_mask_list = [pred_body_countour_binary, pred_soft_mask_binary, pred_bone_mask_binary]

    # compute the pixelwise gradient of the image using sobel filter
    acutance_whole = np.absolute(sobel(synCT_data))
    # normalize CT data and synCT data by clipping to the range [-1024, 3000] and adding 1024
    ct_data = ct_data.clip(min_boundary, max_boundary)
    ct_data += -min_boundary
    synCT_data = synCT_data.clip(min_boundary, max_boundary)
    synCT_data += -min_boundary

    for i in range(len(region_list)):
        region = region_list[i]
        mask = mask_list[i]
        pred_mask = pred_mask_list[i]

        # compute metrics
        mae = np.mean(np.abs(ct_data[mask] - synCT_data[mask]))
        ssim_val = ssim(ct_data[mask], synCT_data[mask], data_range=max_boundary - min_boundary, mask=mask)
        psnr_val = psnr(ct_data[mask], synCT_data[mask], data_range=max_boundary - min_boundary)
        # compute DSC 
        intersection = np.sum(mask & pred_mask)
        union = np.sum(mask | pred_mask)
        dsc = 2 * intersection / (np.sum(mask) + np.sum(pred_mask))
        # compute  acutance by computing the average absolute gradient of the image
        acutance = np.mean( acutance_whole[mask])

        print(f"Computed metrics {region} for {case_name}, MAE: {mae:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}, DSC: {dsc:.4f}, Accutance: { acutance:.4f}")

        # save metrics
        if case_name not in metrics_dict["mae_by_case"]:
            metrics_dict["mae_by_case"][case_name] = {}
            metrics_dict["ssim_by_case"][case_name] = {}
            metrics_dict["psnr_by_case"][case_name] = {}
            metrics_dict["dsc_by_case"][case_name] = {}
            metrics_dict[" acutance_by_case"][case_name] = {}
        metrics_dict["mae_by_case"][case_name][region] = mae
        metrics_dict["ssim_by_case"][case_name][region] = ssim_val
        metrics_dict["psnr_by_case"][case_name][region] = psnr_val
        metrics_dict["dsc_by_case"][case_name][region] = dsc
        metrics_dict[" acutance_by_case"][case_name][region] =  acutance

        if region not in metrics_dict["mae_by_region"]:
            metrics_dict["mae_by_region"][region] = []
            metrics_dict["ssim_by_region"][region] = []
            metrics_dict["psnr_by_region"][region] = []
            metrics_dict["dsc_by_region"][region] = []
            metrics_dict[" acutance_by_region"][region] = []
        metrics_dict["mae_by_region"][region].append(mae)
        metrics_dict["ssim_by_region"][region].append(ssim_val)
        metrics_dict["psnr_by_region"][region].append(psnr_val)
        metrics_dict["dsc_by_region"][region].append(dsc)
        metrics_dict[" acutance_by_region"][region].append( acutance)

# compute average metrics
for region in region_list:
    metrics_dict["mae_by_region"][region] = np.mean(metrics_dict["mae_by_region"][region])
    metrics_dict["ssim_by_region"][region] = np.mean(metrics_dict["ssim_by_region"][region])
    metrics_dict["psnr_by_region"][region] = np.mean(metrics_dict["psnr_by_region"][region])
    metrics_dict["dsc_by_region"][region] = np.mean(metrics_dict["dsc_by_region"][region])
    metrics_dict[" acutance_by_region"][region] = np.mean(metrics_dict[" acutance_by_region"][region])

print("Average metrics by region:")
print(f"MAE: {metrics_dict['mae_by_region']}")
print(f"SSIM: {metrics_dict['ssim_by_region']}")
print(f"PSNR: {metrics_dict['psnr_by_region']}")
print(f"DSC: {metrics_dict['dsc_by_region']}")
print(f"Accutance: {metrics_dict[' acutance_by_region']}")
print("Metrics by case:")
print(f"MAE: {metrics_dict['mae_by_case']}")
print(f"SSIM: {metrics_dict['ssim_by_case']}")
print(f"PSNR: {metrics_dict['psnr_by_case']}")
print(f"DSC: {metrics_dict['dsc_by_case']}")
print(f"Accutance: {metrics_dict[' acutance_by_case']}")

# Save metrics to json
import json
metrics_json_path = f"{root_dir}/James81_metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_dict, f)

print(f"Saved metrics to {metrics_json_path}")
print("Done!")


