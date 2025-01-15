import os
import torch
import monai.transforms
import argparse
import json
import monai
import torch

from monai.utils import set_determinism
from scripts.sample import LDMSampler
from scripts.utils import define_instance



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

def load_segmentation_maps(folder_path):
    """
    Load segmentation maps from a folder.

    Args:
        folder_path (str): Path to the folder containing segmentation maps.

    Returns:
        list: List of loaded segmentation maps.
    """
    segmentation_maps = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            filepath = os.path.join(folder_path, filename)
            segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(filepath)
            segmentation_maps.append(segmentation_map)
    return segmentation_maps

def generate_synthetic_ct_from_maps(ldm_sampler, folder_path):
    """
    Generate synthetic CT images from segmentation maps in a folder.

    Args:
        ldm_sampler (LDMSampler): The LDMSampler instance.
        folder_path (str): Path to the folder containing segmentation maps.

    Returns:
        list: List of generated synthetic CT images.
    """
    segmentation_maps = load_segmentation_maps(folder_path)
    synthetic_images = []
    for segmentation_map in segmentation_maps:
        # Prepare tensors for the segmentation map
        # show segmentation_map shape
        print(f"segmentation_map shape: {segmentation_map.shape}")
        top_region_index_tensor = torch.FloatTensor([79]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor([335]).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
        spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(ldm_sampler.device) * 1e2
        # show top_region_index_tensor, bottom_region_index_tensor, spacing_tensor
        print(f"top_region_index_tensor: {top_region_index_tensor}")
        print(f"bottom_region_index_tensor: {bottom_region_index_tensor}")
        print(f"spacing_tensor: {spacing_tensor}")

        # Generate synthetic image
        synthetic_image, _ = ldm_sampler.sample_one_pair(
            combine_label_or_aug=segmentation_map,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        synthetic_images.append(synthetic_image)
    return synthetic_images

print(f"Everything goes well!")
# Generate synthetic CT images from your segmentation maps
synthetic_images = generate_synthetic_ct_from_maps(ldm_sampler, "Seg2SynCT_test")

# 3dresample -dxyz 1.5 1.5 2.0 -rmode Cu -prefix CTACIVV_E4128_MAISI_conferRes.nii.gz -input CTACIVV_E4128_MAISI.nii.gz
# 3dresample -dxyz 2.734 2.734 2.734 -rmode Cu -prefix CTACIVV_E4128_MAISI_256.nii.gz -input CTACIVV_E4128_MAISI.nii.gz