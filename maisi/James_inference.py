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

case_list_total = [
    "BII096","BPO124","DZS099",
    "EGS066","EIA058","FGX078",
    "FNG137","GSB106","HNJ120",
    "JLB061","JQR130","KQA094",
    "KWX131","KZF084","LBO118",
    "LCQ128","MLU077","NAF069",
    "NIR103","NKQ091","ONC134",
    "OOP125","PAW055","RFK139",
    "RSE114","SAM079","SCH068",
    "SNF129","SPT074","TTE081",
    "TVA105","WLX138","WVX115",
    "XZG098","YKY073","ZTS092",
]
# here ask for the input of case_list division
user_input = input("Please enter the case list division (1-12): ")
case_list_division = int(user_input)
if case_list_division == 1:
    case_list = ["BII096","BPO124","DZS099"]
    device = torch.device("cuda:4")
elif case_list_division == 2:
    case_list = ["EGS066","EIA058","FGX078"]
    device = torch.device("cuda:4")
elif case_list_division == 3:
    case_list = ["FNG137","GSB106","HNJ120"]
    device = torch.device("cuda:4")
elif case_list_division == 4:
    case_list = ["JLB061","JQR130","KQA094"]
    device = torch.device("cuda:4")
elif case_list_division == 5:
    case_list = ["KWX131","KZF084","LBO118"]
    device = torch.device("cuda:5")
elif case_list_division == 6:
    case_list = ["LCQ128","MLU077","NAF069"]
    device = torch.device("cuda:5")
elif case_list_division == 7:
    case_list = ["NIR103","NKQ091","ONC134"]
    device = torch.device("cuda:5")
elif case_list_division == 8:
    case_list = ["OOP125","PAW055","RFK139"]
    device = torch.device("cuda:5")
elif case_list_division == 9:
    case_list = ["RSE114","SAM079","SCH068"]
    device = torch.device("cuda:6")
elif case_list_division == 10:
    case_list = ["SNF129","SPT074","TTE081"]
    device = torch.device("cuda:6")
elif case_list_division == 11:
    case_list = ["TVA105","WLX138","WVX115"]
    device = torch.device("cuda:6")
elif case_list_division == 12:
    case_list = ["XZG098","YKY073","ZTS092"]
    device = torch.device("cuda:6")
else:
    print("Invalid input, please enter a number between 1 and 12.")
    exit()

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
    return synthetic_image.to(device)




work_dir = "James_36"
# ct_dir = f"{work_dir}/ct"
con_dir = f"{work_dir}/body_contour"
synCT_seg_dir = f"{work_dir}/overlap_seg"
synCT_dir = f"{work_dir}/synCT"
os.makedirs(synCT_dir, exist_ok=True)

top_region_index_tensor = torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
spacing_tensor = torch.FloatTensor([1.5, 1.5, 1.5]).to(ldm_sampler.device)  # Example tensor




print("The current device is", device)
txtlog = open(f"{synCT_dir}/log.txt", "w")

for case_name in case_list:
    case_idx = f"E4{case_name[3:]}"
    body_contour_path = f"{work_dir}/mask_body_contour_{case_idx}_Spacing15.nii.gz"
    synCT_seg_path = f"{synCT_seg_dir}/vanila_overlap_{case_idx}_Spacing15.nii.gz"
    synCT_path = f"{synCT_dir}/SynCT_{case_idx}.nii.gz"
    segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(synCT_seg_path)

    top_region_index_tensor = torch.FloatTensor([0, 0, 1, 0]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
    bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
    spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(ldm_sampler.device) * 1e2

    # we use slide_window to inference
    synthetic_image = sliding_window_inference(
        inputs=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
        roi_size=(256, 256, 256),
        sw_batch_size=1,
        predictor=inference_function,
        overlap=1/4,
        mode="gaussian",
        sigma_scale=0.125,
        # device=device,
        # sw_device=device,
        # inputs=segmentation_map.unsqueeze(0).to(ldm_sampler.device),
        # roi_size=(256, 256, 256),
        # sw_batch_size=1,
        # sw_d
        # predictor=inference_function,
        # overlap=1/8,
    )
    
    # save it using ct_path header and affine
    # ct_file = nib.load(ct_path)
    # ct_data = ct_file.get_fdata()
    # ct_header = ct_file.header
    # ct_affine = ct_file.affine
    mask_file = nib.load(body_contour_path)
    mask_data = mask_file.get_fdata()
    synthetic_image = synthetic_image.squeeze().detach().cpu().numpy()
    
    background = mask_data < 0.5
    synthetic_image[background] = -1024
    synthetic_image = nib.Nifti1Image(synthetic_image, mask_file.affine, mask_file.header)
    nib.save(synthetic_image, synCT_path)
    print(f"Synthetic CT image saved to {synCT_path}")

    # # compute mae using ct_data and synthetic_image and apply the mask_data
    # mae = np.abs(ct_data - synthetic_image)
    # synthetic_image = synthetic_image * mask_data
    # mae = mae * mask_data
    # synthetic_image = nib.Nifti1Image(synthetic_image, ct_affine, ct_header)
    # nib.save(synthetic_image, synCT_path)
    # print(f"Synthetic CT image saved to {synCT_path}, MAE: {mae.mean()}")
    # txtlog.write(f"{case_name}: {mae.mean()}\n")
