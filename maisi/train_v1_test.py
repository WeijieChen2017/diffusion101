import os
import time
import json
import argparse

import torch
import nibabel as nib
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from scripts.utils import define_instance
from monai.apps import download_url

from train_v1_utils import create_data_loader
from monai.inferers import sliding_window_inference



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


# we need to run the main function and gettting terminal line input configs

# default_project_name = "cv0_    EncTrue_DecTrue_    epochs300_Lossmae_seed729_    x256_y256_z32"
default_project_name = "cv0_EncFalse_DecTrue_epochs600_LossMAE_seed729_x128_y128_z128"

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
    parser.add_argument("--gpu", type=int, default=4, help="GPU index.")
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

    # Define the autoencoder
    autoencoder = download_and_reload_ckpt()

    # Load the checkpoint
    checkpoint = torch.load(pretrain_weights_path)

    # Load the model's weights
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    print("Pre-trained weights loaded from: ", pretrain_weights_path)

    # Move the model to the desired device
    autoencoder.to(device)

    # Optional: If you want to continue training, restore other components
    # Example:
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # best_val_loss = checkpoint["best_val_loss"]
    # best_val_epoch = checkpoint["best_val_epoch"]

    print("Checkpoint successfully loaded.")

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
        cache_rate=0.1,
        is_inference=True,
    )
    data_loader_train = return_dict["train_loader"]
    data_loader_val = return_dict["val_loader"]
    data_loader_test = return_dict["test_loader"]

    # eval_dict is to perform the evaluation on training/validation/testing datasets and save it to different folders
    eval_dict = {
        "test": data_loader_test,
        # "train": data_loader_train,
        # "val": data_loader_val,
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

    for key, data_loader in eval_dict.items():
        log_str = f"Start testing on {key} dataset."
        log_print(log_file, log_str)

        eval_save_dir = os.path.join(test_result_dir, key)
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_data_loader = data_loader
        average_mae = 0.0

        for i, batch in enumerate(eval_data_loader):
            data_PET = batch["PET"].to(device)
            data_CT = batch["CT"]
            data_mask = batch["BODY"]
            filepath_CT = batch[f"CT_meta_dict"]["filename_or_obj"][0]
            print(f"Test {i+1}: {filepath_CT}")
            filename_CT = os.path.basename(filepath_CT)
            
            with autocast():
                with torch.no_grad():
                    data_synCT = sliding_window_inference(
                        inputs=data_PET,
                        roi_size=(args.dim_x, args.dim_y, args.dim_z),
                        sw_batch_size=args.batchsize,
                        predictor=predictor,
                        # overlap=0.25, 
                        # mode=constant, 
                        # sigma_scale=0.125, 
                        # padding_mode=constant, 
                        # cval=0.0, 
                        # sw_device=None, 
                        # device=None, 
                        # progress=False, 
                        # roi_weight_map=None, 
                        # process_fn=None, 
                        # buffer_steps=None, 
                        # buffer_dim=-1, 
                        # with_coord=False,
                    )
                    
                    # get the synthetic CT data
                    data_synCT = data_synCT.detach().cpu().numpy().squeeze()
                    data_CT = data_CT.detach().cpu().numpy().squeeze()

                    # compute the MAE between the synthetic CT and the ground truth CT
                    masked_data_synCT = data_synCT * data_mask
                    masked_data_CT = data_CT * data_mask
                    abs_diff = np.abs(masked_data_synCT - masked_data_CT)
                    masked_mae = np.sum(abs_diff) / np.sum(data_mask) * 4000 # HU range: -1024 to 2976

                    # load the CT nifti file and take the header and affine to save the synthetic CT
                    CT_nii = nib.load(filepath_CT)
                    CT_affine = CT_nii.affine
                    CT_header = CT_nii.header
                    # save the synthetic CT
                    data_synCT_nii = nib.Nifti1Image(data_synCT, CT_affine, CT_header)
                    filename_synCT = filename_CT.replace("CTACIVV", "synCT")
                    filepath_synCT = os.path.join(eval_save_dir, filename_synCT)
                    nib.save(data_synCT_nii, filepath_synCT)

                    log_str = f"{key} {i+1}: {filename_CT}, MAE: {masked_mae:.4f}, SynCT saved at: {filepath_synCT}."
                    log_print(log_file, log_str)
                    average_mae += masked_mae
        
        average_mae /= len(eval_data_loader)
        log_str = f"Average MAE on {key} dataset: {average_mae:.4f}"
        log_print(log_file, log_str)

if __name__ == "__main__":
    main()




# training log for Jan 7
# current input is 256*256*32, batchsize = 1, GPU memory occupasion = 30543MiB /  40960MiB
# next step we could try 128*128*128
# now on DGX2
# GPU 4, 256*256*32, Train Enc and Dec, "train_1"
# GPU 2, 128*128*128, Train Enc and Dec, "train_2"
# GPU 5, 256*256*32, Train Enc, "eval_1"
# now on DGX1
# GPU 4, 128*128*128, Train Enc, "train_1"
# GPU 3, 256*256*32, Train Dec, "train_2"
# GPU 5, 128*128*128, Train Dec, "train_3"