import os
import time
import json
import argparse

import torch

from scripts.utils import define_instance
from monai.apps import download_url

from train_v1_utils import create_data_loader

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
    # get the boolean value to determine whether encoder is trainable
    parser.add_argument("--train_encoder", type=bool, default=True, help="Train the encoder.")
    # get the boolean value to determine whether decoder is trainable
    parser.add_argument("--train_decoder", type=bool, default=True, help="Train the decoder.")
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
    # get the loss function, default is "mae"
    parser.add_argument("--loss", type=str, default="mae", help="Loss function.")
    # get the random GPU index, default is 4
    parser.add_argument("--gpu", type=int, default=4, help="GPU index.")
    # set the random seed for reproducibility
    parser.add_argument("--seed", type=int, default=729, help="Random seed.")
    
    args = parser.parse_args()
    # apply the random seed
    torch.manual_seed(args.seed)

    # combine the above arguments into a single project name
    project_name = f"cv{args.cv_index}_\
    Enc{args.train_encoder}_Dec{args.train_decoder}_\
    epochs{args.epochs}_Loss{args.loss}_seed{args.seed}_\
    x{args.dim_x}_y{args.dim_y}_z{args.dim_z}"

    # get the project directory
    project_dir = os.path.join(root_dir, project_name)
    # create the project directory
    os.makedirs(project_dir, exist_ok=True)
    # save the configurations to the project directory
    with open(os.path.join(project_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # set the GPU index
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autoencoder = download_and_reload_ckpt()

    return_dict = create_data_loader(
        data_div_json=None,
        cv_index=args.cv_index,
        return_train=True,
        return_val=True,
        return_test=False,
        output_size=(args.dim_x, args.dim_y, args.dim_z),
        batchsize=args.batchsize,
    )
    data_division_dict = return_dict["data_division_dict"]
    data_loader_train = return_dict["train_loader"]
    data_loader_val = return_dict["val_loader"]
    # data_loader_test = data_division_dict["test_loader"]

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
    if args.loss == "mae":
        loss_fn = torch.nn.L1Loss()
    elif args.loss == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    # define the scheduler using StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # define the training loop
    best_val_loss = float("inf")
    best_val_epoch = 0
    save_per_epoch = 20
    autoencoder.to(device)

    for epoch in range(args.epochs):
        autoencoder.train()
        train_loss = 0.0
        for i, batch in enumerate(data_loader_train):
            # in the data loader, the input is a tuple of (input, label, mask)
            data_PET = batch["PET"].to(device).float().type(torch.float32)
            data_CT = batch["CT"].to(device).float().type(torch.float32)
            data_mask = batch["BODY"].to(device).float().type(torch.float32)
            # print the data shape of all three data
            print("data_PET shape: ", data_PET.shape, data_PET.dtype)
            print("data_CT shape: ", data_CT.shape, data_CT.dtype)
            print("data_mask shape: ", data_mask.shape, data_mask.dtype)
            optimizer.zero_grad()
            outputs = autoencoder(data_PET)
            loss = loss_fn(outputs[0], data_CT)
            # apply the boolean mask to the loss
            loss = loss * data_mask
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * 4000 # for denormalization
        train_loss /= len(data_loader_train)

        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(data_loader_val):
                data_PET = batch["PET"].to(device).type(torch.float32)
                data_CT = batch["CT"].to(device).type(torch.float32)
                data_mask = batch["BODY"].to(device).type(torch.float32)
                outputs = autoencoder(data_PET)
                loss = loss_fn(outputs[0], data_CT)
                loss = loss * data_mask
                val_loss += loss.item() * 4000
            val_loss /= len(data_loader_val)

        log_str = f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} at epoch {best_val_epoch}."
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