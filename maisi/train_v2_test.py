import os
import time
import json
import argparse

import torch
import numpy as np
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler

from scripts.utils import define_instance
from monai.apps import download_url
from monai.inferers import sliding_window_inference

from train_v2_utils import create_data_loader

from monai.losses import DiceCELoss

# now train_v2 will output the bone segmentation map

# use DSC, IoU, Hausdoff as the metric
from monai.metrics import DiceMetric, HausdorffDistanceMetric

root_dir = "project"
os.makedirs(root_dir, exist_ok=True)

def log_print(log_file, log_str):
    print(log_str)
    with open(log_file, "a") as f:
        f.write(log_str + "\n")

def download_and_reload_ckpt(directory=None):

    directory = "./"
    if directory is not None:
        os.makedirs(directory, exist_ok=True)

    # TODO: remove the `files` after the files are uploaded to the NGC
    files = [
        {
            "path": "models/autoencoder_epoch273.pt",
            "url": "https://developer.download.nvidia.com/assets/Clara/monai/tutorials"
            "/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
        },
    ]

    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(directory, file["path"])
        
        download_url(url=file["url"], filepath=file["path"])

    print("Downloaded files:")
    for file in files:
        print(file["path"])

    args = argparse.Namespace()

    environment_file = "./configs/environment.json"
    with open(environment_file, "r") as f:
        env_dict = json.load(f)
    for k, v in env_dict.items():
        # Update the path to the downloaded dataset in MONAI_DATA_DIRECTORY
        val = v if "datasets/" not in v else os.path.join(directory, v)
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

    latent_shape = [args.latent_channels, args.output_size[0] // 4, args.output_size[1] // 4, args.output_size[2] // 4]
    print("Network definition and inference inputs have been loaded.")

    autoencoder = define_instance(args, "autoencoder_def")
    checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
    autoencoder.load_state_dict(checkpoint_autoencoder)

    # def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     z_mu, z_sigma = self.encode(x)
    #     z = self.sampling(z_mu, z_sigma)
    #     reconstruction = self.decode(z)
    #     return reconstruction, z_mu, z_sigma

    # print("reconstruction shape: ", output[0].shape)
    # print("z_mu shape: ", output[1].shape)
    # print("z_sigma shape: ", output[2].shape)

    return autoencoder

default_project_name = "cv0_EncTrue_DecTrue_epochs300_LossDiceCELoss_seed729_x256_y256_z32"

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--directory", type=str, default=None, help="Directory to save the downloaded files.")

    # get the project folder 
    parser.add_argument("--project_name", type=str, default=default_project_name, help="project name.")

    # # get the cv index to perform cross-validation
    # parser.add_argument("--cv_index", type=int, default=0, help="Cross-validation index.")
    # # get the boolean value to determine whether encoder is trainable
    # parser.add_argument("--train_encoder", type=bool, default=True, help="Train the encoder.")
    # # get the boolean value to determine whether decoder is trainable
    # parser.add_argument("--train_decoder", type=bool, default=True, help="Train the decoder.")
    # # get the image dim x for each batch
    # parser.add_argument("--dim_x", type=int, default=256, help="Image dimension x.")
    # # get the image dim y for each batch
    # parser.add_argument("--dim_y", type=int, default=256, help="Image dimension y.")
    # # get the image dim z for each batch
    # parser.add_argument("--dim_z", type=int, default=32, help="Image dimension z.")
    # # get the training epochs, default is 300
    # parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    # # get the batchsize
    # parser.add_argument("--batchsize", type=int, default=1, help="Batch size.")
    # # get the loss function, default is "mae"
    # parser.add_argument("--loss", type=str, default="MAE", help="Loss function.")
    # get the random GPU index, default is 0
    parser.add_argument("--gpu", type=int, default=2, help="GPU index.")
    # # set the random seed for reproducibility
    # parser.add_argument("--seed", type=int, default=729, help="Random seed.")
    
    args = parser.parse_args()
    # get the project directory
    project_name = args.project_name
    project_dir = os.path.join(root_dir, project_name)
    # get the configures from the config.json
    config_file = os.path.join(project_dir, "config.json")
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    exclude_keys = ["project_name", "gpu"]
    for k, v in config_dict.items():
        # do not update the project name and gpu index
        if k not in exclude_keys:
            setattr(args, k, v)
        else:
            print(f"{k}: {v} is not updated. Current value: {getattr(args, k)}")
    print("Global config variables have been loaded.")

    # print the project directory
    print(f"Project Directory: {project_dir}")
    # print the configurations
    print("Configurations: ", json.dumps(vars(args), indent=4))

    # set the GPU index
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autoencoder = download_and_reload_ckpt()
    # load the autoencoder model pre-trained weights
    pretrain_weights_path = os.path.join(project_dir, "best_model.pth")
    autoencoder.load_state_dict(torch.load(pretrain_weights_path))
    print("Pre-trained weights loaded at: ", pretrain_weights_path)
    autoencoder.to(device)


    return_dict = create_data_loader(
        data_div_json=None,
        cv_index=args.cv_index,
        return_train=True,
        return_val=True,
        return_test=True,
        output_size=(args.dim_x, args.dim_y, args.dim_z),
        batchsize=args.batchsize,
        num_samples=args.num_samples,
        cache_rate=0.1,
        input_modality = ["PET", "BONE"],
    )
    data_loader_train = return_dict["train_loader"]
    data_loader_val = return_dict["val_loader"]
    data_loader_test = return_dict["test_loader"]

    # eval_dict is to perform the evaluation on training/validation/testing datasets and save it to different folders
    eval_dict = {
        "test": data_loader_test,
        "train": data_loader_train,
        "val": data_loader_val,
    }

    # set the training progress log file in the project directory
    log_file = os.path.join(project_dir, "test_log.txt")
    # write the base configurations to the log file and timestamp
    with open(log_file, "w") as f:
        f.write(f"Project Name: {project_name}\n")
        f.write(f"Project Directory: {project_dir}\n")
        f.write(f"Test Configurations: {json.dumps(vars(args), indent=4)}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
    
    # create the test result directory
    test_result_dir = os.path.join(project_dir, "test_results")
    os.makedirs(test_result_dir, exist_ok=True)
    
    # Define a predictor that extracts only the first tensor
    def predictor(patch_data):
        outputs = autoencoder(patch_data)  # Model output is a tuple of 3 tensors
        return outputs[0]  # Return only the first tensor

    # start the inference
    autoencoder.eval()
    # Initialize metrics once outside the loop
    metric_DSC = DiceMetric(include_background=False)
    metric_Hausdorff = HausdorffDistanceMetric(include_background=False, percentile=95)


    for key, data_loader in eval_dict.items():
        log_str = f"Start testing on {key} dataset."
        log_print(log_file, log_str)

        eval_save_dir = os.path.join(test_result_dir, key)
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_data_loader = data_loader

        test_DSC = 0.0
        test_IoU = 0.0
        test_Hausdorff = 0.0
        for i, batch in enumerate(eval_data_loader):
            data_PET = batch["PET"].to(device)
            data_BONE = batch["BONE"].to(device)
            filepath_BONE = batch[f"BONE_meta_dict"]["filename_or_obj"][0]
            print(f"Test {i+1}: {filepath_BONE}")
            filename_BONE = os.path.basename(filepath_BONE)
            
            with torch.no_grad():
                with autocast():
                    # Perform inference
                    data_synBONE = sliding_window_inference(
                        inputs=data_PET,
                        roi_size=(args.dim_x, args.dim_y, args.dim_z),
                        sw_batch_size=args.batchsize,
                        predictor=predictor,
                    )

                    # Threshold predictions
                    data_synBONE = torch.sigmoid(data_synBONE)  # Probabilities in [0, 1]
                    data_synBONE = (data_synBONE > 0.5).float()  # Binary segmentation

                # Ensure correct shapes and types
                data_synBONE = data_synBONE.unsqueeze(1) if data_synBONE.ndim == 4 else data_synBONE
                data_BONE = data_BONE.unsqueeze(1) if data_BONE.ndim == 4 else data_BONE
                data_synBONE = data_synBONE.contiguous()
                data_BONE = data_BONE.contiguous()

                # Skip metrics if masks are empty
                if data_synBONE.sum() == 0 or data_BONE.sum() == 0:
                    print("Empty segmentation mask detected. Skipping this batch.")
                    continue

                # Compute Dice (on GPU)
                DSC = metric_DSC(data_synBONE, data_BONE).item()
                test_DSC += DSC

                # Compute IoU (on GPU)
                IoU = DSC / (2 - DSC)
                test_IoU += IoU

                # Compute Hausdorff (on CPU)
                data_synBONE_cpu = data_synBONE.cpu().contiguous()
                data_BONE_cpu = data_BONE.cpu().contiguous()
                metric_Hausdorff.reset()
                Hausdorff = metric_Hausdorff(data_synBONE_cpu, data_BONE_cpu).item()
                test_Hausdorff += Hausdorff

                # load the CT nifti file and take the header and affine to save the synthetic CT
                BONE_nii = nib.load(filepath_BONE)
                BONE_affine = BONE_nii.affine
                BONE_header = BONE_nii.header
                # save the synthetic BONE mask
                data_synBONE_nii = nib.Nifti1Image(data_synBONE_cpu.numpy().squeeze(), BONE_affine, BONE_header)
                filename_synBONE = filename_BONE.replace("BONE", "synBONE")
                filepath_synBONE = os.path.join(eval_save_dir, filename_synBONE)
                nib.save(data_synBONE_nii, filepath_synBONE)

                log_str = f"{key} {i+1}: {filename_BONE}, DSC: {DSC:.4f}, IoU: {IoU:.4f}, Hausdorff: {Hausdorff:.4f}."
                log_print(log_file, log_str)
            
            test_DSC /= len(data_loader_test)
            test_IoU /= len(data_loader_test)
            test_Hausdorff /= len(data_loader_test)
            
            log_str = f"Average {key} DSC: {test_DSC:.4f}."
            log_print(log_file, log_str)
            log_str = f"Average {key} IoU: {test_IoU:.4f}."
            log_print(log_file, log_str)
            log_str = f"Average {key} Hausdorff: {test_Hausdorff:.4f}."
            log_print(log_file, log_str)
            
if __name__ == "__main__":
    main()


# Epoch 31/300: Val Loss: 0.6137, Best Val Loss: 0.6367 at epoch 20.
# Epoch 31/300: Test DSC: 0.4695, Test IoU: 0.3080, Test Hausdorff: 51.1075.
# Best Test DSC: 0.4695 at epoch 30.
# Best Test IoU: 0.3080 at epoch 30.
# Best Test Hausdorff: 36.2532 at epoch 20.
# Best model saved at epoch 30 with Val Loss: 0.6137.
# Epoch 32/300: Train Loss: 0.5893.
# Epoch 33/300: Train Loss: 0.5808.
# Epoch 34/300: Train Loss: 0.5945.
# Epoch 35/300: Train Loss: 0.5941.
# Epoch 36/300: Train Loss: nan.
# Epoch 37/300: Train Loss: nan.
# Epoch 38/300: Train Loss: nan.
# Epoch 39/300: Train Loss: nan.
# Epoch 40/300: Train Loss: nan.