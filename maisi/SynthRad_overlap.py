# load mask and seg file, then overlap seg labels on mask

import nibabel as nib
import glob
import os

work_dir = "."
con_dir = f"{work_dir}/mask/"
seg_dir = f"{work_dir}/mr/mr_seg"
overlap_dir = f"{work_dir}/overlap"

os.makedirs(overlap_dir, exist_ok=True)

# 1PA001  1PA024  1PA047  1PA065  1PA091  1PA110  1PA133  1PA150  1PA168  1PA185  1PC018  1PC039  1PC058  1PC080
# 1PA004  1PA025  1PA048  1PA070  1PA093  1PA111  1PA134  1PA151  1PA169  1PA187  1PC019  1PC040  1PC059  1PC082
# 1PA005  1PA026  1PA049  1PA073  1PA094  1PA112  1PA136  1PA152  1PA170  1PA188  1PC022  1PC041  1PC061  1PC084
# 1PA009  1PA028  1PA052  1PA074  1PA095  1PA113  1PA137  1PA154  1PA171  1PC000  1PC023  1PC042  1PC063  1PC085
# 1PA010  1PA029  1PA053  1PA076  1PA097  1PA114  1PA138  1PA155  1PA173  1PC001  1PC024  1PC044  1PC065  1PC088
# 1PA011  1PA030  1PA054  1PA079  1PA098  1PA115  1PA140  1PA156  1PA174  1PC004  1PC027  1PC045  1PC066  1PC092
# 1PA012  1PA031  1PA056  1PA080  1PA100  1PA116  1PA141  1PA157  1PA176  1PC006  1PC029  1PC046  1PC069  1PC093
# 1PA014  1PA035  1PA058  1PA081  1PA101  1PA117  1PA142  1PA159  1PA177  1PC007  1PC032  1PC048  1PC070  1PC095
# 1PA018  1PA038  1PA059  1PA083  1PA105  1PA118  1PA144  1PA161  1PA178  1PC010  1PC033  1PC049  1PC071  1PC096
# 1PA019  1PA040  1PA060  1PA084  1PA106  1PA119  1PA145  1PA163  1PA180  1PC011  1PC035  1PC052  1PC073  1PC097
# 1PA020  1PA041  1PA062  1PA086  1PA107  1PA121  1PA146  1PA164  1PA181  1PC013  1PC036  1PC054  1PC077  1PC098
# 1PA021  1PA044  1PA063  1PA088  1PA108  1PA126  1PA147  1PA165  1PA182  1PC015  1PC037  1PC055  1PC078  nifti
# 1PA022  1PA045  1PA064  1PA090  1PA109  1PA127  1PA148  1PA167  1PA183  1PC017  1PC038  1PC057  1PC079  overview

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

for case_name in case_list:
    con_path = f"{con_dir}/{case_name}_mask_shrink5.nii.gz"
    seg_path = f"{seg_dir}/{case_name}_mr_seg.nii.gz"

    con_file = nib.load(con_path)
    seg_file = nib.load(seg_path)

    con_data = con_file.get_fdata()
    seg_data = seg_file.get_fdata()

    print(f"{case_name}: Original con shape: {con_data.shape}, seg shape: {seg_data.shape}")

    # seg con_data as label 200
    overlap_data = con_data.copy()
    overlap_data[con_data > 0.5] = 200
    overlap_data[seg_data > 0] = seg_data[seg_data > 0]

    overlap_nii = nib.Nifti1Image(overlap_data, con_file.affine, con_file.header)
    overlap_path = f"{overlap_dir}/{case_name}_overlap.nii.gz"

    nib.save(overlap_nii, overlap_path)
    print(f"Saved overlap to {overlap_path}")

