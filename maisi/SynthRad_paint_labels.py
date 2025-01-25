import os
import glob
import nibabel as nib
import numpy as np

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

organ_to_index = {
    "adrenal_gland_left": 9,
    "adrenal_gland_right": 8,
    "aorta": 6,
    "autochthon_left": 104,
    "autochthon_right": 105,
    "brain": 22,
    "colon": 62,
    "duodenum": 13,
    "esophagus": 11,
    "femur_left": 93,
    "femur_right": 94,
    "fibula": 128,
    "gallbladder": 10,
    "gluteus_maximus_left": 98,
    "gluteus_maximus_right": 99,
    "gluteus_medius_left": 100,
    "gluteus_medius_right": 101,
    "gluteus_minimus_left": 102,
    "gluteus_minimus_right": 103,
    "heart": 115,
    "hip_left": 95,
    "hip_right": 96,
    "humerus_left": 87,
    "humerus_right": 88,
    "iliac_artery_left": 58,
    "iliac_artery_right": 59,
    "iliac_vena_left": 60,
    "iliac_vena_right": 61,
    "iliopsoas_left": 106,
    "iliopsoas_right": 107,
    "inferior_vena_cava": 7,
    "intervertebral_discs": 128,
    "kidney_left": 14,
    "kidney_right": 5,
    "liver": 1,
    "lung_left": 29,
    "lung_right": 32,
    "pancreas": 4,
    "portal_vein_and_splenic_vein": 17,
    "prostate": 118,
    "quadriceps_femoris_left": 98,
    "quadriceps_femoris_right": 99,
    "sacrum": 97,
    "sartorius_left": 106,
    "sartorius_right": 107,
    "small_bowel": 19,
    "spinal_cord": 121,
    "spleen": 3,
    "stomach": 12,
    "thigh_medial_compartment_left": 100,
    "thigh_medial_compartment_right": 101,
    "thigh_posterior_compartment_left": 98,
    "thigh_posterior_compartment_right": 99,
    "tibia": 97,
    "urinary_bladder": 15,
    "vertebrae": 33
}


work_dir = "SynthRad_nifit"
seg_folder = f"{work_dir}/mr"
con_dir = f"{work_dir}/mask"
label_painting_folder = f"{work_dir}/label_painting"
os.makedirs(label_painting_folder, exist_ok=True)

for case_name in case_list:
    mr_seg_folder = f"{seg_folder}/{case_name}_mr_seg"
    con_path = f"{con_dir}/{case_name}_mask_shrink5.nii.gz"

    con_file = nib.load(con_path)
    con_data = con_file.get_fdata()

    # copy the con_data and set the label to 200
    label_painting_data = con_data.copy()
    label_painting_data[con_data > 0.5] = 200

    for organ_name in organ_to_index.keys():
        seg_path = f"{mr_seg_folder}/{organ_name}.nii.gz"
        seg_file = nib.load(seg_path)
        seg_data = seg_file.get_fdata()

        # get the value as the segmentation label for the organ
        label = organ_to_index[organ_name]
        label_painting_data[seg_data > 0] = label

    # save the label_painting_data
    label_painting_nifti = nib.Nifti1Image(label_painting_data, con_file.affine, con_file.header)
    label_painting_path = f"{label_painting_folder}/{case_name}_label_painting.nii.gz"
    nib.save(label_painting_nifti, label_painting_path)
    print(f"Saved label_painting to {label_painting_path}")
        