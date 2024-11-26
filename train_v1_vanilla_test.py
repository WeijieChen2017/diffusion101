import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf

import torch.optim as optim
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from train_v1_vanilla_utils import prepare_dataset, train_or_eval_or_test_the_batch, printlog
# from train_v1_vanilla_utils import load_inception_model
from global_config import global_config, set_param, get_param

# <<<<<<<<<<<<<<<<<<<<<<<<<<< running setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
root_dir = "projects/v1_vanilla_step1000"
os.path.exists(root_dir) or os.makedirs(root_dir)
data_division_file = "James_data_v3/cv_list.json"
seeds = 729
base_learning_rate = 1e-4
num_frames = 5

set_param("cv", 0)
set_param("num_frames", num_frames)
set_param("root", root_dir)
set_param("lr", base_learning_rate)
set_param("log_txt_path", os.path.join(root_dir, "log.txt"))

# load data data division
with open(data_division_file, "r") as f:
    data_div = json.load(f)


experiment_config = OmegaConf.load("train_v1_vanilla_config.yaml")
print(experiment_config)
for key in experiment_config.keys():
    set_param(key, experiment_config[key])

# <<<<<<<<<<<<<<<<<<<<<<<<<<< prepare data loader, inceptionV3, and mode
# train_loader, val_loader, _ = prepare_dataset(data_div, invlove_test=False)

# Path to save/load pretrained weights
# PRETRAINED_WEIGHTS_PATH = "inception_v3_weights.pth"
# inceptionV3 = load_inception_model(PRETRAINED_WEIGHTS_PATH).to(device)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,   # number of steps
    # loss_type = 'l1'    # L1 or L2
).to(device)

model_ckpt_path = root_dir+"/best.pth"
if os.path.exists(model_ckpt_path):
    printlog(f"Loading model from {model_ckpt_path}")
    checkpoint = torch.load(model_ckpt_path)
    # show all checkpoint keys
    printlog(f"Checkpoint keys: {checkpoint["state_dict"].keys()}")
    diffusion.load_state_dict(checkpoint["state_dict"])
    best_val_loss = checkpoint["loss"]
    printlog(f"Loaded model from {model_ckpt_path}, with the average val loss {best_val_loss:.6f}.")

num_samples = 8
sampled_batch = diffusion.sample(batch_size = num_samples).cpu().numpy()

save_dir = root_dir+"/samples"
os.makedirs(save_dir, exist_ok=True)
save_name = save_dir+"/sampled_batch.npy"
np.save(save_name, sampled_batch)
printlog(f"Sampled batch has been saved to {save_name}.")
