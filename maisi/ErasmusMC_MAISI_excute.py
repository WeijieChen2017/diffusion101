import os
import torch
import monai.transforms
import argparse
import json
import monai
import numpy as np
import nibabel as nib

from monai.utils import set_determinism
from scripts.sample import LDMSampler
from scripts.utils import define_instance

from monai.data import MetaTensor
from monai.transforms import SaveImage

from monai.inferers import sliding_window_inference

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic CT from MAISI labels')
    parser.add_argument('--maisi_dir', type=str, default='ErasmusMC', help='Directory containing MAISI label files')
    parser.add_argument('--output_dir', type=str, default='ErasmusMC', help='Directory to save synthetic CT images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference')
    parser.add_argument('--case_id', type=str, help='Specific case ID to process (optional)')
    parser.add_argument('--contour_type', type=str, default='both', choices=['pet', 'ct', 'both'], 
                        help='Body contour type to use: pet (bcP), ct (bcC), or both')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model configurations
    root_dir = "."
    config_file = "./configs/config_maisi.json"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    model_args = argparse.Namespace()
    for k, v in config_dict.items():
        setattr(model_args, k, v)

    # Load inference configurations
    config_infer_file = "./configs/config_infer.json"
    with open(config_infer_file, "r") as f:
        config_infer_dict = json.load(f)
    for k, v in config_infer_dict.items():
        setattr(model_args, k, v)
        print(f"{k}: {v}")

    # Load environment configurations
    environment_file = "./configs/environment.json"
    with open(environment_file, "r") as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        val = v if "datasets/" not in v else os.path.join(root_dir, v)
        setattr(model_args, k, val)
    print("Global config variables have been loaded.")

    # Initialize model components
    noise_scheduler = define_instance(model_args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(model_args, "mask_generation_noise_scheduler")

    # Load autoencoder
    autoencoder = define_instance(model_args, "autoencoder_def").to(device)
    checkpoint_autoencoder = torch.load(model_args.trained_autoencoder_path, weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    # Load diffusion UNet
    diffusion_unet = define_instance(model_args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(model_args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=True)
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

    # Load controlnet
    controlnet = define_instance(model_args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(model_args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

    # Load mask generation autoencoder
    mask_generation_autoencoder = define_instance(model_args, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_generation_autoencoder = torch.load(model_args.trained_mask_generation_autoencoder_path, weights_only=True)
    mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

    # Load mask generation diffusion UNet
    mask_generation_diffusion_unet = define_instance(model_args, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_generation_diffusion_unet = torch.load(model_args.trained_mask_generation_diffusion_path, weights_only=True)
    mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
    mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

    print("All the trained model weights have been loaded.")

    # Set up LDMSampler
    latent_shape = [model_args.latent_channels, model_args.output_size[0] // 4, model_args.output_size[1] // 4, model_args.output_size[2] // 4]
    set_determinism(seed=0)
    model_args.random_seed = 0

    ldm_sampler = LDMSampler(
        model_args.body_region,
        model_args.anatomy_list,
        model_args.all_mask_files_json,
        model_args.all_anatomy_size_conditions_json,
        model_args.all_mask_files_base_dir,
        model_args.label_dict_json,
        model_args.label_dict_remap_json,
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
        model_args.mask_generation_latent_shape,
        model_args.output_size,
        model_args.output_dir,
        model_args.controllable_anatomy_size,
        image_output_ext=model_args.image_output_ext,
        label_output_ext=model_args.label_output_ext,
        spacing=model_args.spacing,
        num_inference_steps=model_args.num_inference_steps,
        mask_generation_num_inference_steps=model_args.mask_generation_num_inference_steps,
        random_seed=model_args.random_seed,
        autoencoder_sliding_window_infer_size=model_args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=model_args.autoencoder_sliding_window_infer_overlap,
    )

    # Define inference function for sliding window
    def inference_function(inputs):
        top_region_index_tensor = torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).half().to(device) * 1e2
        bottom_region_index_tensor = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0).half().to(device) * 1e2
        spacing_tensor = torch.FloatTensor(ldm_sampler.spacing).unsqueeze(0).half().to(device) * 1e2
        
        synthetic_image, _, _ = ldm_sampler.sample_one_pair(
            combine_label_or_aug=inputs,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        return synthetic_image.to(device)

    # Create log file
    log_file = open(os.path.join(args.output_dir, "inference_log.txt"), "w")
    
    # Determine which body contour versions to use
    contour_suffixes = []
    if args.contour_type.lower() == 'pet':
        contour_suffixes = ['bcP']
    elif args.contour_type.lower() == 'ct':
        contour_suffixes = ['bcC']
    else:  # 'both'
        contour_suffixes = ['bcP', 'bcC']
    
    print(f"Processing body contour types: {contour_suffixes}")
    
    # Process each contour type
    for contour_suffix in contour_suffixes:
        print(f"\nProcessing {contour_suffix} ({'PET' if contour_suffix == 'bcP' else 'CT'} body contour) files")
        log_file.write(f"\nProcessing {contour_suffix} ({'PET' if contour_suffix == 'bcP' else 'CT'} body contour) files\n")
        
        # Process MAISI label files
        maisi_files = []
        if args.case_id:
            # Process only the specified case
            for filename in os.listdir(args.maisi_dir):
                if args.case_id in filename and f"_MAISI_{contour_suffix}.nii.gz" in filename:
                    maisi_files.append(filename)
        else:
            # Process all MAISI files with the specified contour type
            for filename in os.listdir(args.maisi_dir):
                if f"_MAISI_{contour_suffix}.nii.gz" in filename:
                    maisi_files.append(filename)
        
        print(f"Found {len(maisi_files)} MAISI files to process")
        
        for maisi_file in maisi_files:
            # Extract case ID from filename (remove _MAISI_bcX.nii.gz)
            case_id = maisi_file.replace(f"_MAISI_{contour_suffix}.nii.gz", "")
            print(f"Processing case: {case_id}")
            log_file.write(f"Processing case: {case_id}\n")
            
            # Load MAISI segmentation
            maisi_path = os.path.join(args.maisi_dir, maisi_file)
            segmentation_map = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True)(maisi_path)
            
            # Generate synthetic CT using sliding window inference
            print(f"Running inference for {case_id}...")
            try:
                synthetic_image = sliding_window_inference(
                    inputs=segmentation_map.unsqueeze(0).to(device),
                    roi_size=(256, 256, 256),
                    sw_batch_size=1,
                    predictor=inference_function,
                    overlap=1/4,
                    mode="gaussian",
                    sigma_scale=0.125,
                    device=device,
                    sw_device=device
                )
            except RuntimeError as e:
                print(f"Error during inference: {e}")
                log_file.write(f"Error during inference for {case_id}: {e}\n")
                continue  # Continue with next file instead of raising
            
            # Save synthetic CT
            synCT_path = os.path.join(args.output_dir, f"SynCT_{case_id}_{contour_suffix}.nii.gz")
            
            # Get original image metadata
            maisi_img = nib.load(maisi_path)
            
            # Apply body contour mask (class 200) to set background to -1024 HU
            synthetic_image = synthetic_image.squeeze().detach().cpu().numpy()
            maisi_data = maisi_img.get_fdata()
            background = (maisi_data != 200) & (maisi_data == 0)  # Background is where there's no body contour and no tissue
            synthetic_image[background] = -1024
            
            # Save as NIfTI
            synthetic_image = nib.Nifti1Image(synthetic_image, maisi_img.affine, maisi_img.header)
            nib.save(synthetic_image, synCT_path)
            print(f"Synthetic CT image saved to {synCT_path}")
            log_file.write(f"Synthetic CT image saved to {synCT_path}\n")
    
    log_file.close()
    print("All processing completed.")

if __name__ == "__main__":
    main()
