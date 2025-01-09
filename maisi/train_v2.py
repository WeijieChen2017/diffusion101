import os
import time
import json
import argparse

import torch
import numpy as np
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


# we need to run the main function and gettting terminal line input configs

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--directory", type=str, default=None, help="Directory to save the downloaded files.")

    # get the cv index to perform cross-validation
    parser.add_argument("--cv_index", type=int, default=0, help="Cross-validation index.")
    # get the boolean value to determine whether encoder is trainable using store false
    # which means if not using --train_encoder, this should be false
    # parser.add_argument("--train_encoder", type=bool, default=True, help="Train the encoder.")
    parser.add_argument("--train_encoder", dest="train_encoder", action="store_true")
    # get the boolean value to determine whether decoder is trainable
    # parser.add_argument("--train_decoder", type=bool, default=True, help="Train the decoder.")
    parser.add_argument("--train_decoder", dest="train_decoder", action="store_true")
    # get the image dim x for each batch
    parser.add_argument("--dim_x", type=int, default=256, help="Image dimension x.")
    # get the image dim y for each batch
    parser.add_argument("--dim_y", type=int, default=256, help="Image dimension y.")
    # get the image dim z for each batch
    parser.add_argument("--dim_z", type=int, default=32, help="Image dimension z.")
    # get the training epochs, default is 300
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    # get the batchsize
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size.")
    # get the number of samples
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples.")
    # get the loss function, default is "DiceCELoss"
    parser.add_argument("--loss", type=str, default="DiceCELoss", help="Loss function.")
    # get the random GPU index, default is 4
    parser.add_argument("--gpu", type=int, default=2, help="GPU index.")
    # set the random seed for reproducibility
    parser.add_argument("--seed", type=int, default=426, help="Random seed.")
    
    args = parser.parse_args()
    # apply the random seed
    torch.manual_seed(args.seed)

    # combine the above arguments into a single project name
    project_name = f"cv{args.cv_index}_"
    project_name = project_name + f"Enc{args.train_encoder}_Dec{args.train_decoder}_"
    project_name = project_name + f"epochs{args.epochs}_Loss{args.loss}_seed{args.seed}_"
    project_name = project_name + f"x{args.dim_x}_y{args.dim_y}_z{args.dim_z}"

    # get the project directory
    project_dir = os.path.join(root_dir, project_name)
    print(f"Project Name: {project_name}")
    # create the project directory
    os.makedirs(project_dir, exist_ok=True)
    # save the configurations to the project directory
    with open(os.path.join(project_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    # exit() # for checking the project name

    # set the GPU index
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autoencoder = download_and_reload_ckpt()

    # assert num_samples is greater than 1
    assert args.num_samples > 1, "Number of samples must be greater than 1."

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
    data_division_dict = return_dict["data_division_dict"]
    data_loader_train = return_dict["train_loader"]
    data_loader_val = return_dict["val_loader"]
    data_loader_test = return_dict["test_loader"]

    # save the data_division_dict to the project directory
    with open(os.path.join(project_dir, "data_division_dict.json"), "w") as f:
        json.dump(data_division_dict, f, indent=4)

    # set the training progress log file in the project directory
    log_file = os.path.join(project_dir, "train_log.txt")
    # write the base configurations to the log file and timestamp
    with open(log_file, "w") as f:
        f.write(f"Project Name: {project_name}\n")
        f.write(f"Project Directory: {project_dir}\n")
        f.write(f"Training Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configurations: {json.dumps(vars(args), indent=4)}\n")
        f.write("\n")

    # freeze the encoder if the encoder is not trainable
    if not args.train_encoder:
        for param in autoencoder.encoder.parameters():
            param.requires_grad = False
    
    # freeze the decoder if the decoder is not trainable
    if not args.train_decoder:
        for param in autoencoder.decoder.parameters():
            param.requires_grad = False

    # define the optimizer using AdamW
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-4)

    # define the loss function according to the input argument
    if args.loss == "DiceCELoss":
        # Define the loss function
        loss_fn = DiceCELoss(
            include_background=False,  # Ignore background in the loss computation
            sigmoid=True,  # Apply sigmoid to logits for binary segmentation
            lambda_dice=1.0,  # Weight for Dice Loss
            lambda_ce=1.0,  # Weight for Cross-Entropy Loss
        )
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    # define the scheduler using StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Define a predictor that extracts only the first tensor
    def predictor(patch_data):
        outputs = autoencoder(patch_data)  # Model output is a tuple of 3 tensors
        return outputs[0]  # Return only the first tensor

    # define the training loop
    best_val_loss = float("inf")
    best_val_epoch = 0
    save_per_epoch = 20
    eval_per_epoch = 10
    best_test_metric_dict = {
        "DSC": 0.0,
        "IoU": 0.0,
        "Hausdorff": float("inf"),
        "DSC_epoch": 0,
        "IoU_epoch": 0,
        "Hausdorff_epoch": 0,
    }
    autoencoder.to(device)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        autoencoder.train()
        train_loss = 0.0
        for i, batch in enumerate(data_loader_train):
            data_samples_PET = batch["PET"].to(device)
            data_samples_BONE = batch["BONE"].to(device)
                # print the data shape of all three data
                # print("data_PET shape: ", data_PET.shape, data_PET.dtype)
                # print("data_CT shape: ", data_CT.shape, data_CT.dtype)
                # print("data_mask shape: ", data_mask.shape, data_mask.dtype)
            
            for idx_sample in range(args.num_samples):
                optimizer.zero_grad()
                data_PET = data_samples_PET[idx_sample].unsqueeze(0)
                data_BONE = data_samples_BONE[idx_sample].unsqueeze(0)
                with autocast():
                    outputs, _, _ = autoencoder(data_PET)
                    loss = loss_fn(outputs, data_BONE)
                    loss = loss.mean()  # Ensure loss is a scalar

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

        train_loss /= len(data_loader_train)
        train_loss /= args.num_samples
        log_str = f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}."
        log_print(log_file, log_str)

        if epoch % eval_per_epoch == 0:
            autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(data_loader_val):
                    data_samples_PET = batch["PET"].to(device)
                    data_samples_BONE = batch["BONE"].to(device)
                
                    for idx_sample in range(args.num_samples):
                        data_PET = data_samples_PET[idx_sample].unsqueeze(0)
                        data_BONE = data_samples_BONE[idx_sample].unsqueeze(0)
                        with autocast():
                            outputs, _, _ = autoencoder(data_PET)
                            loss = loss_fn(outputs, data_BONE)
                            loss = loss.mean()
                        
                        val_loss += loss.item()
                val_loss /= len(data_loader_val)
                val_loss /= args.num_samples

            log_str = f"Epoch {epoch+1}/{args.epochs}: Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} at epoch {best_val_epoch}."
            log_print(log_file, log_str)

            # do testing
            autoencoder.eval()
            test_DSC = 0.0
            test_IoU = 0.0
            test_Hausdorff = 0.0
            for i, batch in enumerate(data_loader_test):
                data_PET = batch["PET"].to(device)
                data_BONE = batch["BONE"]
                
                with autocast():
                    with torch.no_grad():
                        data_synBONE = sliding_window_inference(
                            inputs=data_PET,
                            roi_size=(args.dim_x, args.dim_y, args.dim_z),
                            sw_batch_size=args.batchsize,
                            predictor=predictor,
                        )
                        
                        # get the synthetic CT data
                        data_synBONE = data_synBONE.detach().cpu().numpy().squeeze()
                        data_BONE = data_BONE.detach().cpu().numpy().squeeze()
                        data_synBONE = np.where(data_synBONE > 0.5, 1, 0)

                        # compute the metrics
                        # DSC
                        metric_DSC = DiceMetric(include_background=False)
                        DSC = metric_DSC(data_synBONE, data_BONE)
                        test_DSC += DSC

                        # IoU
                        IoU = DSC / (2 - DSC)
                        test_IoU += IoU

                        # Hausdorff
                        metric_Hausdorff = HausdorffDistanceMetric(include_background=False, percentile=95)
                        Hausdorff = metric_Hausdorff(data_synBONE, data_BONE)
                        test_Hausdorff += Hausdorff
            
            test_DSC /= len(data_loader_test)
            test_IoU /= len(data_loader_test)
            test_Hausdorff /= len(data_loader_test)
            log_str = f"Epoch {epoch+1}/{args.epochs}: Test DSC: {test_DSC:.4f}, Test IoU: {test_IoU:.4f}, Test Hausdorff: {test_Hausdorff:.4f}."
            log_print(log_file, log_str)
            
            # save the best test metric
            if test_DSC > best_test_metric_dict["DSC"]:
                best_test_metric_dict["DSC"] = test_DSC
                best_test_metric_dict["DSC_epoch"] = epoch
            if test_IoU > best_test_metric_dict["IoU"]:
                best_test_metric_dict["IoU"] = test_IoU
                best_test_metric_dict["IoU_epoch"] = epoch
            if test_Hausdorff < best_test_metric_dict["Hausdorff"]:
                best_test_metric_dict["Hausdorff"] = test_Hausdorff
                best_test_metric_dict["Hausdorff_epoch"] = epoch

            log_str = f"Best Test DSC: {best_test_metric_dict['DSC']:.4f} at epoch {best_test_metric_dict['DSC_epoch']}."
            log_print(log_file, log_str)
            log_str = f"Best Test IoU: {best_test_metric_dict['IoU']:.4f} at epoch {best_test_metric_dict['IoU_epoch']}."
            log_print(log_file, log_str)
            log_str = f"Best Test Hausdorff: {best_test_metric_dict['Hausdorff']:.4f} at epoch {best_test_metric_dict['Hausdorff_epoch']}."
            log_print(log_file, log_str)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(autoencoder.state_dict(), os.path.join(project_dir, "best_model.pth"))
            log_str = f"Best model saved at epoch {epoch} with Val Loss: {val_loss:.4f}."
            log_print(log_file, log_str)

        if epoch % save_per_epoch == 0:
            torch.save(autoencoder.state_dict(), os.path.join(project_dir, f"model_epoch{epoch}.pth"))
            log_str = f"Model saved at epoch {epoch}."
            log_print(log_file, log_str)

        scheduler.step()

    log_str = f"Training completed. Best Val Loss: {best_val_loss:.4f} at epoch {best_val_epoch}."
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
