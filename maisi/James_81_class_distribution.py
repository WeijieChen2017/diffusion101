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

import os
import nibabel as nib
import numpy as np


root_dir = "NAC_CTAC_Spacing15"
synCT_dir = "NAC_CTAC_Spacing15/inference_20250128_noon"

for case_name in case_name_list:

    ct_path = f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"
    maisi_label_path = f"{root_dir}/CTAC_{case_name}_TS_label.nii.gz"
    synCT_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI.nii.gz"

    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    maisi_label_file = nib.load(maisi_label_path)
    maisi_label_data = maisi_label_file.get_fdata()
    synCT_file = nib.load(synCT_path)
    synCT_data = synCT_file.get_fdata()

    # we will load the maisi labels as the key, 
    # for each key, save the distribution of the synCT and CT for 200 bins

    HU_distribution_dict = {}

    maisi_label_values = np.unique(maisi_label_data)
    for seg_label in maisi_label_values:
        if seg_label == 0:
            continue
        seg_mask = maisi_label_data == seg_label
        seg_synCT_values = synCT_data[seg_mask]
        seg_ct_values = ct_data[seg_mask]

        # save the distribution of the synCT and CT for 200 bins
        synCT_hist, synCT_bin_edges = np.histogram(seg_synCT_values, bins=200)
        ct_hist, ct_bin_edges = np.histogram(seg_ct_values, bins=200)

        # save the histogram to the a dict
        HU_distribution_dict[seg_label] = {
            "synCT_hist": synCT_hist,
            "synCT_bin_edges": synCT_bin_edges,
            "ct_hist": ct_hist,
            "ct_bin_edges": ct_bin_edges,
            "num_voxels": np.sum(seg_mask),
        }

    # save the dict to a npy file
    save_path = f"{root_dir}/CTAC_{case_name}_HU_distribution.npy"
    np.save(save_path, HU_distribution_dict)

    print(f"Saved HU distribution to {save_path}")

