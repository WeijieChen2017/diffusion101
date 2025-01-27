import os
import nibabel as nib
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

work_dir = "SynthRad_nifti"
ct_dir = f"{work_dir}/ct"
con_dir = f"{work_dir}/mask/"
seg_dir = f"{work_dir}/mr/mr_seg"
overlap_dir = f"{work_dir}/label_painting"
synCT_dir = f"{work_dir}/synCT_label_painting"

case_list = [
    "1PA001", "1PA024", "1PA047", "1PA065", "1PA091", "1PA110", "1PA133", "1PA150", "1PA168", "1PA185", "1PC018", "1PC039", "1PC058", "1PC080",
    "1PA004", "1PA025", "1PA048", "1PA070", "1PA093", "1PA111", "1PA134", "1PA151", "1PA169", "1PA187", "1PC019", "1PC040", "1PC059", "1PC082",
    "1PA005", "1PA026", "1PA049", "1PA073", "1PA094", "1PA112", "1PA136", "1PA152", "1PA170", "1PA188", "1PC022", "1PC041", "1PC061", "1PC084",
    "1PA009", "1PA028", "1PA052", "1PA074", "1PA095", "1PA113", "1PA137", "1PA154", "1PA171", "1PC000", "1PC023", "1PC042", "1PC063", "1PC085",
    "1PA010", "1PA029", "1PA053", "1PA076", "1PA097", "1PA114", "1PA138", "1PA155", "1PA173", "1PC001", "1PC024", "1PC044", "1PC065", "1PC088",
    "1PA011", "1PA030", "1PA054", "1PA079", "1PA098", "1PA115", "1PA140", "1PA156", "1PA174", "1PC004", "1PC027", "1PC045", "1PC066", "1PC092",
    "1PA012", "1PA031", "1PA056", "1PA080", "1PA100", "1PA116", "1PA141", "1PA157", "1PA176", "1PC006", "1PC029", "1PC046", "1PC069", "1PC093",
    "1PA014", "1PA035", "1PA058", "1PA081", "1PA101", "1PA117", "1PA142", "1PA159", "1PA177", "1PC007", "1PC032", "1PC048", "1PC070", "1PC095",
    "1PA018", "1PA038", "1PA059", "1PA083", "1PA105", "1PA118", "1PA144", "1PA161", "1PA178", "1PC010", "1PC033", "1PC049", "1PC071", "1PC096",
    "1PA019", "1PA040", "1PA060", "1PA084", "1PA106", "1PA119", "1PA145", "1PA163", "1PA180", "1PC011", "1PC035", "1PC052", "1PC073", "1PC097",
    "1PA020", "1PA041", "1PA062", "1PA086", "1PA107", "1PA121", "1PA146", "1PA164", "1PA181", "1PC013", "1PC036", "1PC054", "1PC077", "1PC098",
    "1PA021", "1PA044", "1PA063", "1PA088", "1PA108", "1PA126", "1PA147", "1PA165", "1PA182", "1PC015", "1PC037", "1PC055", "1PC078",
    "1PA022", "1PA045", "1PA064", "1PA090", "1PA109", "1PA127", "1PA148", "1PA167", "1PA183", "1PC017", "1PC038", "1PC057", "1PC079",
]

metric_mae = 0
metric_ssim = 0
metric_psnr = 0

metric_mae_mid = 0
metric_ssim_mid = 0
metric_psnr_mid = 0

for case_name in case_list:
    overlap_path = f"{overlap_dir}/{case_name}_label_painting.nii.gz"
    con_path = f"{con_dir}/{case_name}_mask_shrink5.nii.gz"
    metric_mask_path = f"{con_dir}/{case_name}_mask.nii.gz"
    synCT_path = f"{synCT_dir}/{case_name}_label_painting.nii.gz"
    ct_path = f"{ct_dir}/{case_name}_ct.nii.gz"

    ct_file = nib.load(ct_path)
    con_file = nib.load(con_path)
    mask_file = nib.load(metric_mask_path)
    synCT_file = nib.load(synCT_path)

    ct_data = ct_file.get_fdata()
    con_data = con_file.get_fdata()
    mask_data = mask_file.get_fdata()
    synCT_data = synCT_file.get_fdata()

    minHU, maxHU = -1024, 3000

    # set background to -1024
    synCT_bg = synCT_data.copy()
    synCT_bg[con_data < 0.5] = -1024
    synCT_bg_file = nib.Nifti1Image(synCT_bg, synCT_file.affine, synCT_file.header)
    synCT_bg_path = f"{synCT_dir}/{case_name}_bg.nii.gz"
    nib.save(synCT_bg_file, synCT_bg_path)

    # compute metrics of synCT_bg and ct
    nonneg_ct = ct_data - minHU
    nonneg_synCT_bg = synCT_bg - minHU
    # clip
    nonneg_ct[nonneg_ct < 0] = 0
    nonneg_ct[nonneg_ct > maxHU - minHU] = maxHU - minHU
    nonneg_synCT_bg[nonneg_synCT_bg < 0] = 0
    nonneg_synCT_bg[nonneg_synCT_bg > maxHU - minHU] = maxHU - minHU

    # MAE based on body contour con_data
    # case_mae = np.mean(np.abs(nonneg_synCT_bg[con_data > 0.5] - nonneg_ct[con_data > 0.5]))
    # case_mae_mid = np.mean(np.abs(nonneg_synCT_bg[con_data > 0.5][:, :, 1:-1] - nonneg_ct[con_data > 0.5][:, :, 1:-1]))
    # metric_mae += case_mae
    # metric_mae_mid += case_mae_mid

    # SSIM and PSNRbased on body contour con_data [-1024, 3000]
    len_z = con_data.shape[2]
    case_mae = 0
    case_ssim = 0
    case_psnr = 0
    case_mae_mid = 0
    case_ssim_mid = 0
    case_psnr_mid = 0
    for z in range(len_z):
        masked_ct = nonneg_ct[:, :, z][mask_data[:, :, z] > 0.5]
        masked_synCT_bg = nonneg_synCT_bg[:, :, z][mask_data[:, :, z] > 0.5]
        case_mae += np.mean(np.abs(masked_ct - masked_synCT_bg))
        case_ssim += ssim(masked_ct, masked_synCT_bg, data_range=maxHU - minHU)
        case_psnr += psnr(masked_ct, masked_synCT_bg, data_range=maxHU - minHU)
        if z > 0 and z < len_z - 1:
            case_mae_mid += np.mean(np.abs(masked_ct - masked_synCT_bg))
            case_ssim_mid += ssim(masked_ct, masked_synCT_bg, data_range=maxHU - minHU)
            case_psnr_mid += psnr(masked_ct, masked_synCT_bg, data_range=maxHU - minHU)

    case_mae /= len_z
    case_ssim /= len_z
    case_psnr /= len_z
    case_mae_mid /= (len_z - 2)
    case_ssim_mid /= (len_z - 2)
    case_psnr_mid /= (len_z - 2)

    metric_mae += case_mae
    metric_ssim += case_ssim
    metric_psnr += case_psnr
    metric_mae_mid += case_mae_mid
    metric_ssim_mid += case_ssim_mid
    metric_psnr_mid += case_psnr_mid

    print(f"Processed {case_name}: MAE {case_mae:.4f}, SSIM {case_ssim:.4f}, PSNR {case_psnr:.4f}, MAE mid {case_mae_mid:.4f}, SSIM mid {case_ssim_mid:.4f}, PSNR mid {case_psnr_mid:.4f}")

metric_mae /= len(case_list)
metric_ssim /= len(case_list)
metric_psnr /= len(case_list)
metric_mae_mid /= len(case_list)
metric_ssim_mid /= len(case_list)
metric_psnr_mid /= len(case_list)

print(f"Overall: MAE {metric_mae:.4f}, SSIM {metric_ssim:.4f}, PSNR {metric_psnr:.4f}, MAE mid {metric_mae_mid:.4f}, SSIM mid {metric_ssim_mid:.4f}, PSNR mid {metric_psnr_mid:.4f}")




