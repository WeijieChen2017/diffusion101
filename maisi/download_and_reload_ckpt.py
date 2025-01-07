import os
from monai.apps import download_url

directory = "./"
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

# TODO: remove the `files` after the files are uploaded to the NGC
files = [
    {
        "path": "models/autoencoder_epoch273.pt",
        "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
        "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
    },
]

for file in files:
    file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
    download_url(url=file["url"], filepath=file["path"])

print("Downloaded files:")
for file in files:
    print(file["path"])

import argparse
import json

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

# check_input(
#     args.body_region,
#     args.anatomy_list,
#     args.label_dict_json,
#     args.output_size,
#     args.spacing,
#     args.controllable_anatomy_size,
# )
latent_shape = [args.latent_channels, args.output_size[0] // 4, args.output_size[1] // 4, args.output_size[2] // 4]
print("Network definition and inference inputs have been loaded.")

from scripts.utils import define_instance
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

autoencoder = define_instance(args, "autoencoder_def").to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
autoencoder.load_state_dict(checkpoint_autoencoder)

# for k, v in checkpoint_autoencoder.items():
#     print(k)

# create a 128*128*128 random tensors in the shape of 1*1*128*128*128

random_tensor = torch.randn(1, 1, 256, 256, 256).to(device).float()
print(random_tensor.shape)

# Cast the input tensor to half precision (float16)
random_tensor = random_tensor.type(torch.float16)  

# Cast the model's weights to float16
autoencoder = autoencoder.half() # or autoencoder.to(torch.float16)

output = autoencoder(random_tensor)
# print(output.shape)

# def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     z_mu, z_sigma = self.encode(x)
#     z = self.sampling(z_mu, z_sigma)
#     reconstruction = self.decode(z)
#     return reconstruction, z_mu, z_sigma

print("reconstruction shape: ", output[0].shape)
print("z_mu shape: ", output[1].shape)
print("z_sigma shape: ", output[2].shape)

