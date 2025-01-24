import os
import torch
import monai.transforms
import argparse
import json
import monai
import torch
import numpy as np
import nibabel as nib

from monai.utils import set_determinism
from scripts.sample import LDMSampler
from scripts.utils import define_instance

from datetime import datetime

from monai.data import MetaTensor
from monai.transforms import SaveImage

from monai.inferers import sliding_window_inference


root_dir = "."
save_dir = os.path.join(root_dir, "inference_v1")
os.makedirs(save_dir, exist_ok=True)

args = argparse.Namespace()

config_file = "./configs/config_maisi.json"
with open(config_file, "r") as f:
    config_dict = json.load(f)
for k, v in config_dict.items():
    setattr(args, k, v)

# check the format of inference inputs
config_infer_file = "./configs/config_infer.json"
with open(config_infer_file, "r") as f:
    config_infer_dict = json.load(f)
for k, v in config_infer_dict.items():
    setattr(args, k, v)
    print(f"{k}: {v}")

environment_file = "./configs/environment.json"
with open(environment_file, "r") as f:
    env_dict = json.load(f)
for k, v in env_dict.items():
    # Update the path to the downloaded dataset in MONAI_DATA_DIRECTORY
    val = v if "datasets/" not in v else os.path.join(root_dir, v)
    setattr(args, k, val)
    print(f"{k}: {val}")
print("Global config variables have been loaded.")

noise_scheduler = define_instance(args, "noise_scheduler")
mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

device = torch.device("cuda:0")

autoencoder = define_instance(args, "autoencoder_def").to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
autoencoder.load_state_dict(checkpoint_autoencoder)

diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=True)
scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

controlnet = define_instance(args, "controlnet_def").to(device)
checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
checkpoint_mask_generation_autoencoder = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
checkpoint_mask_generation_diffusion_unet = torch.load(args.trained_mask_generation_diffusion_path, weights_only=True)
mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

print("All the trained model weights have been loaded.")

latent_shape = [args.latent_channels, args.output_size[0] // 4, args.output_size[1] // 4, args.output_size[2] // 4]
set_determinism(seed=0)
args.random_seed = 0

# Initialize the LDMSampler with appropriate parameters
ldm_sampler = LDMSampler(
        args.body_region,
        args.anatomy_list,
        args.all_mask_files_json,
        args.all_anatomy_size_conditions_json,
        args.all_mask_files_base_dir,
        args.label_dict_json,
        args.label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        args.mask_generation_latent_shape,
        args.output_size,
        args.output_dir,
        args.controllable_anatomy_size,
        image_output_ext=args.image_output_ext,
        label_output_ext=args.label_output_ext,
        spacing=args.spacing,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
)

# def load_segmentation_maps(folder_path):
#     """
#     Load segmentation maps from a folder.

#     Args:
#         folder_path (str): Path to the folder containing segmentation maps.

#     Returns:
#         list: List of loaded segmentation maps.
#     """
#     segmentation_maps = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".nii.gz"):
#             filepath = os.path.join(folder_path, filename)
#             segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(filepath)
#             segmentation_maps.append(segmentation_map)
#     return segmentation_maps

# def generate_synthetic_ct_from_maps(ldm_sampler, folder_path):
#     """
#     Generate synthetic CT images from segmentation maps in a folder.

#     Args:
#         ldm_sampler (LDMSampler): The LDMSampler instance.
#         folder_path (str): Path to the folder containing segmentation maps.

#     Returns:
#         list: List of generated synthetic CT images.
#     """
#     segmentation_maps = load_segmentation_maps(folder_path)
#     synthetic_images = []

#     # segmentation_map shape: torch.Size([1, 256, 256, 401])
#     # top_region_index_tensor: tensor([[7900.]], device='cuda:0', dtype=torch.float16)
#     # bottom_region_index_tensor: tensor([[33504.]], device='cuda:0', dtype=torch.float16)
#     # spacing_tensor: tensor([[150., 150., 200.]], device='cuda:0', dtype=torch.float16)

#     for i, segmentation_map in enumerate(segmentation_maps):
#         # Prepare tensors for the segmentation map
#         # show segmentation_map shape
#         # print(f"segmentation_map shape: {segmentation_map.shape}") # 
#         top_region_index_tensor = torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
#         bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
#         spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
#         # show top_region_index_tensor, bottom_region_index_tensor, spacing_tensor
#         # print(f"top_region_index_tensor: {top_region_index_tensor}")
#         # print(f"bottom_region_index_tensor: {bottom_region_index_tensor}")
#         # print(f"spacing_tensor: {spacing_tensor}")

#         # Generate synthetic image
#         synthetic_image, _ = ldm_sampler.sample_one_pair(
#             combine_label_or_aug=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
#             top_region_index_tensor=top_region_index_tensor,
#             bottom_region_index_tensor=bottom_region_index_tensor,
#             spacing_tensor=spacing_tensor,
#         )
#         # synthetic_images.append(synthetic_image)
#         # take the filepath 
#         filepath = fo
    # return synthetic_images

def generate_and_save_synthetic_ct(ldm_sampler, folder_path):
    """
    Load segmentation maps from a folder, generate synthetic CT images, and save them with a new name ending with _synCT.nii.gz.

    Args:
        ldm_sampler (LDMSampler): The LDMSampler instance.
        folder_path (str): Path to the folder containing segmentation maps.
    """
    # Load segmentation maps
    segmentation_maps = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            filepath = os.path.join(folder_path, filename)
            segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(filepath)
            # segmentation_maps.append((segmentation_map, filepath))

            # Generate synthetic CT images and save them
            # for segmentation_map, filepath in segmentation_maps:
            # Prepare tensors for the segmentation map
            top_region_index_tensor = torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
            bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
            spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(ldm_sampler.device) * 1e2

            # # Generate synthetic image
            # synthetic_image, syntheic_mask, combine_label_or_aug = ldm_sampler.sample_one_pair(
            #     combine_label_or_aug=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
            #     top_region_index_tensor=top_region_index_tensor,
            #     bottom_region_index_tensor=bottom_region_index_tensor,
            #     spacing_tensor=spacing_tensor,
            # )

            # # Save the synthetic image with a new name ending with _synCT.nii.gz
            # save_path = filepath.replace(".nii.gz", "_synCT.npy")
            # synthetic_image_numpy = synthetic_image.detach().cpu().numpy()
            # np.save(save_path, synthetic_image_numpy)
            # print(f"Synthetic CT image saved to {save_path}, shape: {synthetic_image_numpy.shape}")
            # # save_image = SaveImage(output_dir=os.path.dirname(save_path), output_postfix="_synCT", output_ext=".nii.gz", separate_folder=False)
            # # save_image(synthetic_image[0], save_path)

            # Generate synthetic image
            synthetic_image, synthetic_mask, _ = ldm_sampler.sample_one_pair(
                combine_label_or_aug=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )

            # Generate unique timestamp-based postfix
            output_postfix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Save synthetic CT image in .npy format for debugging/reference
            # npy_save_path = filepath.replace(".nii.gz", f"_{output_postfix}_synCT.npy")
            # synthetic_image_numpy = synthetic_image.detach().cpu().numpy()
            # np.save(npy_save_path, synthetic_image_numpy)
            # print(f"Synthetic CT image saved to {npy_save_path}, shape: {synthetic_image_numpy.shape}")

            # Save synthetic CT image in .nii.gz format
            # save_path = filepath.replace(".nii.gz", f"_{output_postfix}_synCT.nii.gz")
            save_path = filepath.replace("cube", f"synCT_cube")
            synthetic_image = MetaTensor(synthetic_image, meta=segmentation_map.meta)  # Use metadata from segmentation map
            img_saver = SaveImage(
                output_dir=os.path.dirname(save_path),
                output_postfix="_synCT",
                output_ext=".nii.gz",
                separate_folder=False,
            )
            img_saver(synthetic_image[0])
            print(f"Synthetic CT image saved to {save_path}")

            # Save synthetic label
            # synthetic_label_save_path = filepath.replace(".nii.gz", f"_{output_postfix}_synLabel.nii.gz")
            synthetic_label_save_path = filepath.replace("cube", f"synLabel_cube")
            synthetic_mask = MetaTensor(synthetic_mask, meta=segmentation_map.meta)
            label_saver = SaveImage(
                output_dir=os.path.dirname(synthetic_label_save_path),
                output_postfix="_synLabel",
                output_ext=".nii.gz",
                separate_folder=False,
            )
            label_saver(synthetic_mask[0])
            print(f"Synthetic label saved to {synthetic_label_save_path}")

            # Save combined label
            # combine_label_save_path = filepath.replace(".nii.gz", f"_{output_postfix}_combineLabel.nii.gz")
            # combine_label_or_aug = MetaTensor(combine_label_or_aug, meta=segmentation_map.meta)
            # combine_label_saver = SaveImage(
            #     output_dir=os.path.dirname(combine_label_save_path),
            #     output_postfix="_combineLabel",
            #     output_ext=".nii.gz",
            #     separate_folder=False,
            # )
            # combine_label_saver(combine_label_or_aug[0])
            # print(f"Combined label saved to {combine_label_save_path}")

# print(f"Everything goes well at loading the trained model weights and setting up the LDMSampler instance.")
# Generate synthetic CT images from your segmentation maps
# synthetic_images = generate_and_save_synthetic_ct(ldm_sampler, "NAC_synCT_MAISI_xy5")

# here we wrap model to let it be a function for sliding_window_inference
def inference_function(inputs):
    synthetic_image, _, _ = ldm_sampler.sample_one_pair(
        combine_label_or_aug=inputs,
        top_region_index_tensor=top_region_index_tensor,
        bottom_region_index_tensor=bottom_region_index_tensor,
        spacing_tensor=spacing_tensor,
    )
    return synthetic_image


work_dir = "."
ct_dir = f"{work_dir}/ct"
con_dir = f"{work_dir}/mask/"
seg_dir = f"{work_dir}/mr/mr_seg"
overlap_dir = f"{work_dir}/overlap"
synCT_dir = f"{work_dir}/synCT"
top_region_index_tensor = torch.FloatTensor([0, 0, 1, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
spacing_tensor = torch.FloatTensor([1.5, 1.5, 1.5]).to(ldm_sampler.device)  # Example tensor

os.makedirs(synCT_dir, exist_ok=True)

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
    overlap_path = f"{overlap_dir}/{case_name}_overlap.nii.gz"
    mask_path = f"{con_dir}/{case_name}_mask.nii.gz"
    synCT_path = f"{synCT_dir}/{case_name}_synCT.nii.gz"
    ct_path = f"{ct_dir}/{case_name}_ct.nii.gz"
    segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(overlap_path)

    top_region_index_tensor = torch.FloatTensor([0, 0, 1, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
    bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
    spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(ldm_sampler.device) * 1e2

    # we use slide_window to inference
    synthetic_image = sliding_window_inference(
        inputs=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
        roi_size=(256, 256, 256),
        sw_batch_size=1,
        predictor=inference_function,
        overlap=1/8,
    )
    
    # save it using ct_path header and affine
    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    ct_header = ct_file.header
    ct_affine = ct_file.affine
    mask_file = nib.load(mask_path)
    mask_data = mask_file.get_fdata()
    synthetic_image = synthetic_image.detach().cpu().numpy()

    # compute mae using ct_data and synthetic_image and apply the mask_data
    mae = np.abs(ct_data - synthetic_image)
    synthetic_image = synthetic_image * mask_data
    mae = mae * mask_data
    synthetic_image = nib.Nifti1Image(synthetic_image, ct_affine, ct_header)
    nib.save(synthetic_image, synCT_path)
    print(f"Synthetic CT image saved to {synCT_path}, MAE: {mae.mean()}")
