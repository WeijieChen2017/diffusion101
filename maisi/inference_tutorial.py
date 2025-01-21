import argparse
import json
import os
import tempfile

import monai
import torch
import nibabel as nib
import numpy as np
from monai.apps import download_url
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism
from scripts.sample import LDMSampler, check_input
from scripts.utils import define_instance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image

print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = "." if directory is None else directory

# TODO: remove the `files` after the files are uploaded to the NGC
files = [
    {
        "path": "models/autoencoder_epoch273.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
        "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
    },
    {
        "path": "models/input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
        "/model_maisi_input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1_alternative.pt",
    },
    {
        "path": "models/controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo"
        "/model_maisi_controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current_alternative.pt",
    },
    {
        "path": "models/mask_generation_autoencoder.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai" "/tutorials/mask_generation_autoencoder.pt",
    },
    {
        "path": "models/mask_generation_diffusion_unet.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai"
        "/tutorials/model_zoo/model_maisi_mask_generation_diffusion_unet_v2.pt",
    },
    {
        "path": "configs/candidate_masks_flexible_size_and_spacing_3000.json",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai"
        "/tutorials/candidate_masks_flexible_size_and_spacing_3000.json",
    },
    {
        "path": "configs/all_anatomy_size_condtions.json",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/all_anatomy_size_condtions.json",
    },
    {
        "path": "datasets/all_masks_flexible_size_and_spacing_3000.zip",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai"
        "/tutorials/model_zoo/model_maisi_all_masks_flexible_size_and_spacing_3000.zip",
    },
]

for file in files:
    file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
    download_url(url=file["url"], filepath=file["path"])

args = argparse.Namespace()

environment_file = "./configs/environment.json"
with open(environment_file, "r") as f:
    env_dict = json.load(f)
for k, v in env_dict.items():
    # Update the path to the downloaded dataset in MONAI_DATA_DIRECTORY
    val = v if "datasets/" not in v else os.path.join(root_dir, v)
    setattr(args, k, val)
    print(f"{k}: {val}")
print("Global config variables have been loaded.")

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

check_input(
    args.body_region,
    args.anatomy_list,
    args.label_dict_json,
    args.output_size,
    args.spacing,
    args.controllable_anatomy_size,
)
latent_shape = [args.latent_channels, args.output_size[0] // 4, args.output_size[1] // 4, args.output_size[2] // 4]
print("Network definition and inference inputs have been loaded.")

set_determinism(seed=0)
args.random_seed = 0

noise_scheduler = define_instance(args, "noise_scheduler")
mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

device = torch.device("cuda:1")

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

print(f"The generated image/mask pairs will be saved in {args.output_dir}.")
output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
print("MAISI image/mask generation finished")

# visualize_image_filename = output_filenames[0][0]
# visualize_mask_filename = output_filenames[0][1]
# print(f"Visualizing {visualize_image_filename} and {visualize_mask_filename}...")

# # load image/mask pairs
# loader = LoadImage(image_only=True, ensure_channel_first=True)
# orientation = Orientation(axcodes="RAS")
# image_volume = orientation(loader(visualize_image_filename))
# mask_volume = orientation(loader(visualize_mask_filename)).to(torch.uint8)

# # visualize for CT HU intensity between [-200, 500]
# image_volume = torch.clip(image_volume, -200, 500)
# image_volume = image_volume - torch.min(image_volume)
# image_volume = image_volume / torch.max(image_volume)

# # save the mask and image to the output directory as nifti files
# nifti_image_path = os.path.join(args.output_dir, "visualize_image.nii.gz")
# nifti_mask_path = os.path.join(args.output_dir, "visualize_mask.nii.gz")

# nifti_image_npy = image_volume.cpu().numpy()
# nifti_mask_npy = mask_volume.cpu().numpy()
# np.save(nifti_image_path.replace(".nii.gz", ".npy"), nifti_image_npy)
# np.save(nifti_mask_path.replace(".nii.gz", ".npy"), nifti_mask_npy)


# nifti_image_file = nib.Nifti1Image(nifti_image_npy, affine=None)
# nib.save(nifti_image_file, nifti_image_path)
# nifti_mask_file = nib.Nifti1Image(nifti_mask_npy, affine=None)
# nib.save(nifti_mask_file, nifti_mask_path)

# print(f"Visualized image saved to {nifti_image_path} and mask saved to {nifti_mask_path}")
