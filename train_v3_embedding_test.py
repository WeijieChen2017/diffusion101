import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf

import torch.optim as optim
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from train_v3_embedding_utils import prepare_dataset, printlog, test_diffusion_model_and_save_slices
# from train_v1_vanilla_utils import load_inception_model
from global_config import global_config, set_param, get_param

# <<<<<<<<<<<<<<<<<<<<<<<<<<< running setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cv_folds = 0
root_dir = f"projects/v3_img_petCond_acs_cv{cv_folds}"
os.path.exists(root_dir) or os.makedirs(root_dir)
data_division_file = "James_data_v3/cv_list.json"
seeds = 729
base_learning_rate = 1e-4
num_frames = 5
batch_size = 18
sampling_timesteps = 50

set_param("cv", 0)
set_param("num_frames", num_frames)
set_param("root", root_dir)
set_param("lr", base_learning_rate)
set_param("log_txt_path", os.path.join(root_dir, "log.txt"))

# load data data division
with open(data_division_file, "r") as f:
    data_div = json.load(f)


experiment_config = OmegaConf.load("train_v3_embedding_config.yaml")
print(experiment_config)
for key in experiment_config.keys():
    set_param(key, experiment_config[key])

# <<<<<<<<<<<<<<<<<<<<<<<<<<< prepare data loader, inceptionV3, and mode
_, _, test_loader = prepare_dataset(
    data_div, 
    invlove_train=False, 
    invlove_val=False, 
    invlove_test=True
)

# Path to save/load pretrained weights
# PRETRAINED_WEIGHTS_PATH = "inception_v3_weights.pth"
# inceptionV3 = load_inception_model(PRETRAINED_WEIGHTS_PATH).to(device)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 6,
    flash_attn = False,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,   # number of steps
    sampling_timesteps = sampling_timesteps, # for ddim sampling
    # loss_type = 'l1'    # L1 or L2
).to(device)

model_ckpt_path = root_dir+"/best.pth"
if os.path.exists(model_ckpt_path):
    printlog(f"Loading model from {model_ckpt_path}")
    checkpoint = torch.load(model_ckpt_path)
    ckpt_keys = checkpoint["state_dict"].keys()
    # show all checkpoint keys
    # printlog(f"Checkpoint keys, {ckpt_keys}")
    diffusion.load_state_dict(checkpoint["state_dict"])
    best_val_loss = checkpoint["loss"]
    printlog(f"Loaded model from {model_ckpt_path}, with the average val loss {best_val_loss:.6f}.")

# num_samples = 8
# sampled_batch = diffusion.sample(batch_size = num_samples).cpu().numpy()

# Assume `cond` is a tensor of shape (batch_size, cond_channels, h, w)
# Normalize or preprocess `cond` if needed (depending on how the model expects it)

# Pass `cond` to the sampling method
# sampled_batch = diffusion.sample(batch_size=num_samples, cond=cond).cpu().numpy()

# Load VQ weights
vq_weights = np.load("James_data_v3/vq_f4_weights_attn.npy")  # You'll need to specify the correct path

# Test the model and save results
output_directory = root_dir+f"/test_results_ddim_batch_{batch_size}_step_{sampling_timesteps}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
test_diffusion_model_and_save_slices(
    data_loader=test_loader,
    model=diffusion,
    device=device, 
    output_dir=output_directory,
    vq_weights=vq_weights,
    batch_size=batch_size,
)
