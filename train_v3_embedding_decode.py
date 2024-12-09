data_div_file = "James_data_v3/cv_list.json"
cv = 0
results_folder = f"projects/v3_img_petCond_acs_cv{cv}/test_results_ddpm_batch_128/"
model_pretrain_weights = "vq_f4.pth"

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from train_v3_embedding_decode_utils import nnVQModel

model_step1_params = {
    "VQ_NAME": "f4",
    "n_embed": 8192,
    "embed_dim": 3,
    "img_size" : 256,
    "input_modality" : ["TOFNAC", "CTAC"],
    "ckpt_path": "model.ckpt",
    "ddconfig": {
        "double_z": False,
        "z_channels": 3,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
}

nnmodel = nnVQModel(
    ddconfig=model_step1_params["ddconfig"],
    n_embed=model_step1_params["n_embed"],
    embed_dim=model_step1_params["embed_dim"],
    ckpt_path=None,
    ignore_keys=[],
    image_key="image",
)

# Load pretrained weights
if os.path.exists(model_pretrain_weights):
    nnmodel.init_from_ckpt(model_pretrain_weights)
else:
    print(f"Warning: Pretrained weights file '{model_pretrain_weights}' not found. Using randomly initialized weights.")

