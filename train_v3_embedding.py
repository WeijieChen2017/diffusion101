import os
import json
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F

import torch.optim as optim
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

from train_v3_embedding_utils import prepare_dataset, train_or_eval_or_test_the_batch_cond, printlog, get_loss_function
# from train_v1_vanilla_utils import load_inception_model
from global_config import global_config, set_param, get_param

# <<<<<<<<<<<<<<<<<<<<<<<<<<< running setting
experiment_config = OmegaConf.load("train_v3_embedding_config.yaml")
print(experiment_config)
for key in experiment_config.keys():
    set_param(key, experiment_config[key])

# DGX2
# 3 - cv0
# 5 - cv1
# 6 - cv2

# GPU2
# 0 - cv3
# 1 - cv4
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cv_folds = 4
loss_type = get_param("train_param")["loss_type"]
root_dir = f"projects/v3_emb_petCond_acs_cv{cv_folds}_{loss_type}_sphere"
os.path.exists(root_dir) or os.makedirs(root_dir)
data_division_file = "James_data_v3/cv_list.json"
seeds = 729
base_learning_rate = 1e-4

set_param("cv", cv_folds)
set_param("root", root_dir)
set_param("lr", base_learning_rate)
set_param("log_txt_path", os.path.join(root_dir, "log_train.txt"))

# load data data division
with open(data_division_file, "r") as f:
    data_div = json.load(f)

# <<<<<<<<<<<<<<<<<<<<<<<<<<< prepare data loader, inceptionV3, and mode
train_loader, val_loader, _ = prepare_dataset(
    data_div, 
    invlove_train=True,
    invlove_val=True,
    invlove_test=False,
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
    # loss_type = 'l1'    # L1 or L2
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate)

# After loading the model and before the training loop
loss_fn = get_loss_function(get_param("train_param")["loss_type"])

# Training and validation loop
best_val_loss = float("inf")
best_val_epoch = 0
epoch = get_param("train_param")["epoch"]
for idx_epoch in range(epoch):

    printlog(f"Epoch [{idx_epoch}]/[{epoch}]")

    # ===============training stage===============

    model.train()
    loss_1st = 0.0
    loss_2nd = 0.0
    loss_3rd = 0.0
    total_case_train = len(train_loader)

    for idx_case, batch in enumerate(train_loader):
        cl_1, cl_2, cl_3 = train_or_eval_or_test_the_batch_cond(
            batch=batch,
            batch_size=get_param("train_param")["train_stage"]["batch_size"],
            stage="train",
            model=diffusion,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        loss_1st += cl_1
        loss_2nd += cl_2
        loss_3rd += cl_3
        printlog(f"<Train> Epoch [{idx_epoch}]/[{epoch}], Case [{idx_case}]/[{total_case_train}], Loss 1st {cl_1:.6f}, Loss 2nd {cl_2:.6f}, Loss 3rd {cl_3:.6f}")

    loss_1st /= len(train_loader)
    loss_2nd /= len(train_loader)
    loss_3rd /= len(train_loader)
    avg_loss = (loss_1st + loss_2nd + loss_3rd) / 3
    printlog(f"<Train> Epoch [{idx_epoch}]/[{epoch}], Loss 1st {loss_1st:.6f}, Loss 2nd {loss_2nd:.6f}, Loss 3rd {loss_3rd:.6f}, Avg Loss {avg_loss:.6f}")

    # ===============validation stage===============
    model.eval()
    loss_1st = 0.0
    loss_2nd = 0.0
    loss_3rd = 0.0
    total_case_val = len(val_loader)

    for idx_case, batch in enumerate(val_loader):
        cl_1, cl_2, cl_3 = train_or_eval_or_test_the_batch_cond(
            batch=batch,
            batch_size=get_param("train_param")["val_stage"]["batch_size"],
            stage="eval",
            model=diffusion,
            device=device,
            loss_fn=loss_fn,
        )
        loss_1st += cl_1
        loss_2nd += cl_2
        loss_3rd += cl_3
        printlog(f"<Val> Epoch [{idx_epoch}]/[{epoch}], Case [{idx_case}]/[{total_case_val}], Loss 1st {cl_1:.6f}, Loss 2nd {cl_2:.6f}, Loss 3rd {cl_3:.6f}")

    loss_1st /= len(val_loader)
    loss_2nd /= len(val_loader)
    loss_3rd /= len(val_loader)
    avg_loss = (loss_1st + loss_2nd + loss_3rd) / 3
    printlog(f"<Val> Epoch [{idx_epoch}]/[{epoch}], Loss 1st {loss_1st:.6f}, Loss 2nd {loss_2nd:.6f}, Loss 3rd {loss_3rd:.6f}, Avg Loss {avg_loss:.6f}")
    printlog(f"The best val loss is {best_val_loss:.6f} at epoch {best_val_epoch}")
    
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        best_val_epoch = idx_epoch
        torch.save({
            "state_dict": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": idx_epoch,
            "loss": avg_loss,
        }, os.path.join(root_dir, "best.pth"))
        printlog(f"Best model saved at epoch {idx_epoch}, with the average val loss {avg_loss:.6f}.")
    
    if idx_epoch % get_param("train_param")["save_per_epoch"] == 0:
        torch.save({
            "state_dict": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": idx_epoch,
            "loss": avg_loss,
        }, os.path.join(root_dir, f"epoch_{idx_epoch}.pth"))
        printlog(f"Model saved at epoch {idx_epoch}")

