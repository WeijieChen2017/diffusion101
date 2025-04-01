from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
)
import torch
import argparse
import random
import os
import numpy as np
import time
import datetime
from LDM_utils import prepare_dataset, DataConfig, DataLoaderConfig, VQModel

HU_MIN = -1024
HU_MAX = 1976

def normalize_residual(residual, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Normalize the residual to [-1, 1] range
    """
    range_hu = hu_max - hu_min
    return 2.0 * residual / range_hu

def denormalize_residual(normalized_residual, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Denormalize the residual from [-1, 1] range back to HU range
    """
    range_hu = hu_max - hu_min
    return normalized_residual * range_hu / 2.0

def setup_logger(log_dir):
    """
    Set up a logger that writes to a text file
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    def log_message(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Print to console
        print(log_entry)
        
        # Write to log file
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
    
    return log_message

def train_or_eval(train_or_eval, model, volume_x, volume_y, optimizer, output_loss, LOSS_FACTOR):
    
    # 1, z, 256, 256 tensor
    axial_case_loss = 0
    coronal_case_loss = 0
    sagittal_case_loss = 0

    # if z is not divided by 4, pad the volume_x and volume_y
    if volume_x.shape[1] % 4 != 0:
        pad_size = 4 - volume_x.shape[1] % 4
        # The torch.nn.functional.pad function uses a specific order for padding dimensions, which is: (left, right, top, bottom, front, back) for 4D tensors. This means that when padding in 4D, the padding should apply as follows:
        # The first two values (left, right) correspond to the last dimension.
        # The next two values (top, bottom) correspond to the second-to-last dimension.
        # The final two values (front, back) correspond to the third-to-last dimension.
        volume_x = torch.nn.functional.pad(volume_x, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
        volume_y = torch.nn.functional.pad(volume_y, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)

    indices_list_axial = [i for i in range(1, volume_x.shape[1]-1)]
    indices_list_coronal = [i for i in range(1, volume_x.shape[2]-1)]
    indices_list_sagittal = [i for i in range(1, volume_x.shape[3]-1)]
    random.shuffle(indices_list_axial)
    random.shuffle(indices_list_coronal)
    random.shuffle(indices_list_sagittal)

    if train_or_eval == "train":
        # axial slices
        for indices in indices_list_axial:
            # Input context: 3 adjacent slices
            x = volume_x[:, indices-1:indices+2, :, :]
            
            # Target: single slice
            y_target = volume_y[:, indices, :, :].unsqueeze(1)  # 1, 1, 256, 256
            # Input center slice for residual addition
            x_center = volume_x[:, indices, :, :].unsqueeze(1)  # 1, 1, 256, 256
        
            optimizer.zero_grad()
            # Model predicts residual
            residual = model(x)
            # Clip residual to [-1, 1] range
            residual = torch.clamp(residual, -1.0, 1.0)
            # Final prediction = input + residual
            prediction = x_center + residual
            # Loss between prediction and target
            loss = output_loss(prediction, y_target)
            loss.backward()
            optimizer.step()

            axial_case_loss += loss.item()

        axial_case_loss /= len(indices_list_axial)
        axial_case_loss *= LOSS_FACTOR

        # coronal slices
        for indices in indices_list_coronal:
            x = volume_x[:, :, indices-1:indices+2, :] # 1, z, 3, 256
            # convert it to 1, 3, z, 256
            x = x.permute(0, 2, 1, 3)
            
            # Target: single slice
            y_target = volume_y[:, :, indices, :].unsqueeze(1) # 1, 1, z, 256
            # Input center slice for residual addition
            x_center = volume_x[:, :, indices, :].unsqueeze(1) # 1, 1, z, 256

            optimizer.zero_grad()
            # Model predicts residual
            residual = model(x)
            # Clip residual to [-1, 1] range
            residual = torch.clamp(residual, -1.0, 1.0)
            # Final prediction = input + residual
            prediction = x_center + residual
            # Loss between prediction and target
            loss = output_loss(prediction, y_target)
            loss.backward()
            optimizer.step()

            coronal_case_loss += loss.item()
        
        coronal_case_loss /= len(indices_list_coronal)
        coronal_case_loss *= LOSS_FACTOR

        # sagittal slices
        for indices in indices_list_sagittal:
            x = volume_x[:, :, :, indices-1:indices+2] # 1, z, 256, 3
            x = x.permute(0, 3, 1, 2) # 1, 3, z, 256
            
            # Target: single slice
            y_target = volume_y[:, :, :, indices].unsqueeze(1)
            # Input center slice for residual addition
            x_center = volume_x[:, :, :, indices].unsqueeze(1)

            optimizer.zero_grad()
            # Model predicts residual
            residual = model(x)
            # Clip residual to [-1, 1] range
            residual = torch.clamp(residual, -1.0, 1.0)
            # Final prediction = input + residual
            prediction = x_center + residual
            # Loss between prediction and target
            loss = output_loss(prediction, y_target)
            loss.backward()
            optimizer.step()

            sagittal_case_loss += loss.item()
        
        sagittal_case_loss /= len(indices_list_sagittal)
        sagittal_case_loss *= LOSS_FACTOR

    elif train_or_eval == "val":

        # axial slices
        for indices in indices_list_axial:
            x = volume_x[:, indices-1:indices+2, :, :]
            
            # Target: single slice
            y_target = volume_y[:, indices, :, :].unsqueeze(1)
            # Input center slice for residual addition
            x_center = volume_x[:, indices, :, :].unsqueeze(1)

            with torch.no_grad():
                # Model predicts residual
                residual = model(x)
                # Clip residual to [-1, 1] range
                residual = torch.clamp(residual, -1.0, 1.0)
                # Final prediction = input + residual
                prediction = x_center + residual
                # Loss between prediction and target
                loss = output_loss(prediction, y_target)
                axial_case_loss += loss.item()
        
        axial_case_loss /= len(indices_list_axial)
        axial_case_loss *= LOSS_FACTOR

        # coronal slices
        for indices in indices_list_coronal:
            x = volume_x[:, :, indices-1:indices+2, :]
            x = x.permute(0, 2, 1, 3)
            
            # Target: single slice
            y_target = volume_y[:, :, indices, :].unsqueeze(1)
            # Input center slice for residual addition
            x_center = volume_x[:, :, indices, :].unsqueeze(1)

            with torch.no_grad():
                # Model predicts residual
                residual = model(x)
                # Clip residual to [-1, 1] range
                residual = torch.clamp(residual, -1.0, 1.0)
                # Final prediction = input + residual
                prediction = x_center + residual
                # Loss between prediction and target
                loss = output_loss(prediction, y_target)
                coronal_case_loss += loss.item()
        
        coronal_case_loss /= len(indices_list_coronal)
        coronal_case_loss *= LOSS_FACTOR

        # sagittal slices
        for indices in indices_list_sagittal:
            x = volume_x[:, :, :, indices-1:indices+2]
            x = x.permute(0, 3, 1, 2)
            
            # Target: single slice
            y_target = volume_y[:, :, :, indices].unsqueeze(1)
            # Input center slice for residual addition
            x_center = volume_x[:, :, :, indices].unsqueeze(1)

            with torch.no_grad():
                # Model predicts residual
                residual = model(x)
                # Clip residual to [-1, 1] range
                residual = torch.clamp(residual, -1.0, 1.0)
                # Final prediction = input + residual
                prediction = x_center + residual
                # Loss between prediction and target
                loss = output_loss(prediction, y_target)
                sagittal_case_loss += loss.item()
        
        sagittal_case_loss /= len(indices_list_sagittal)
        sagittal_case_loss *= LOSS_FACTOR

    return axial_case_loss, coronal_case_loss, sagittal_case_loss


def examine_data():
    # Create data config
    data_config = DataConfig(
        root_folder="LDM_adapter",
        cross_validation="fold_1",  # Using the exact key from folds.json
        input_modality=["CT", "sCT"],
        train=DataLoaderConfig(
            batch_size=1,
            shuffle=True,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=0.25
        ),
        val=DataLoaderConfig(
            batch_size=1,
            shuffle=False,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=0.5
        ),
        test=DataLoaderConfig(
            batch_size=1,
            shuffle=False,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=0.1
        )
    )
    
    # Create train, val, and test dataloaders using the prepare_dataset function
    train_loader, val_loader, test_loader = prepare_dataset("LDM_adapter/LDM_folds_with_test.json", data_config)
    
    # Get first batch and examine data
    first_batch = next(iter(train_loader))
    ct_image = first_batch["CT"]
    sct_image = first_batch["sCT"]
    
    print("Training data:")
    print(f"Input (sCT) shape: {sct_image.shape}")
    print(f"Input (sCT) min value: {sct_image.min().item()}")
    print(f"Input (sCT) max value: {sct_image.max().item()}")
    print(f"Target (CT) shape: {ct_image.shape}")
    print(f"Target (CT) min value: {ct_image.min().item()}")
    print(f"Target (CT) max value: {ct_image.max().item()}")
    
    # Examine validation data
    val_batch = next(iter(val_loader))
    val_ct = val_batch["CT"]
    val_sct = val_batch["sCT"]
    
    print("\nValidation data:")
    print(f"Input (sCT) shape: {val_sct.shape}")
    print(f"Input (sCT) min value: {val_sct.min().item()}")
    print(f"Input (sCT) max value: {val_sct.max().item()}")
    print(f"Target (CT) shape: {val_ct.shape}")
    print(f"Target (CT) min value: {val_ct.min().item()}")
    print(f"Target (CT) max value: {val_ct.max().item()}")
    
    # Examine test data
    test_batch = next(iter(test_loader))
    test_ct = test_batch["CT"]
    test_sct = test_batch["sCT"]
    
    print("\nTest data:")
    print(f"Input (sCT) shape: {test_sct.shape}")
    print(f"Input (sCT) min value: {test_sct.min().item()}")
    print(f"Input (sCT) max value: {test_sct.max().item()}")
    print(f"Target (CT) shape: {test_ct.shape}")
    print(f"Target (CT) min value: {test_ct.min().item()}")
    print(f"Target (CT) max value: {test_ct.max().item()}")
    
    return train_loader, val_loader, test_loader


def train():
    # Parse arguments
    argparser = argparse.ArgumentParser(description='Train latent diffusion model')
    argparser.add_argument('--cross_validation', type=int, default=1, help='Index of the cross validation fold')
    argparser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    argparser.add_argument('--val_interval', type=int, default=5, help='Validate every N epochs')
    argparser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    argparser.add_argument('--log_dir', type=str, default='LDM_adapter/logs', help='Directory to save logs')
    argparser.add_argument('--checkpoint', type=str, default='LDM_adapter/f4_noattn.pth', 
                          help='Path to pretrained checkpoint')
    args = argparser.parse_args()
    
    # Set random seed for reproducibility
    random_seed = 729
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup logger
    log_dir = args.log_dir
    logger = setup_logger(log_dir)
    logger(f"Starting training with fold {args.cross_validation}")
    logger(f"Random seed: {random_seed}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"Using device: {device}")
    
    # Create folders for saving results
    cross_validation = args.cross_validation
    root_folder = f"./results/fold_{cross_validation}/"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    logger(f"Results will be saved to: {root_folder}")
    
    # Get data loaders
    logger("Loading data...")
    train_loader, val_loader, test_loader = examine_data()
    logger(f"Train dataset size: {len(train_loader)}")
    logger(f"Validation dataset size: {len(val_loader)}")
    logger(f"Test dataset size: {len(test_loader)}")
    
    # Model parameters
    model_params = {
        "VQ_NAME": "ldm-residual-model",
        "n_embed": 8192,
        "embed_dim": 3,
        "img_size": 256,
        "ckpt_path": args.checkpoint,
        "ddconfig": {
            "attn_type": "none",
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,  # Input has 3 channels (context slices)
            "out_ch": 1,       # Output has 1 channel (residual)
            "ch": 128,         # Base channel count
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
    }
    logger(f"Model configuration: {model_params['VQ_NAME']}")
    logger(f"Checkpoint path: {model_params['ckpt_path']}")
    
    # Initialize model
    logger("Initializing model...")
    model = VQModel(
        ddconfig=model_params["ddconfig"],
        n_embed=model_params["n_embed"],
        embed_dim=model_params["embed_dim"],
    )
    
    # Load model from checkpoint if provided
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')), strict=False)
        logger(f"Model weights loaded from {args.checkpoint}")
    except Exception as e:
        logger(f"Could not load model weights from {args.checkpoint}: {str(e)}")
        logger("Initializing model with random weights")
        model.init_random_weights()
    
    model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    logger("Optimizer: AdamW (lr=1e-5, weight_decay=1e-5)")
    
    # Set up loss function
    loss_fn = torch.nn.L1Loss()
    logger("Loss function: L1Loss")
    
    # Range for loss scaling
    LOSS_FACTOR = HU_MAX - HU_MIN  # Equivalent to RANGE_CT
    logger(f"Loss scaling factor: {LOSS_FACTOR}")
    
    # Training loop
    best_val_loss = float('inf')
    
    logger("Starting training loop...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Training phase
        axial_train_loss = 0
        coronal_train_loss = 0
        sagittal_train_loss = 0
        
        for idx, batch in enumerate(train_loader):
            volume_x = batch["sCT"].to(device)
            volume_y = batch["CT"].to(device)
            
            axial_loss, coronal_loss, sagittal_loss = train_or_eval(
                "train", model, volume_x, volume_y, optimizer, loss_fn, LOSS_FACTOR
            )
            
            axial_train_loss += axial_loss
            coronal_train_loss += coronal_loss
            sagittal_train_loss += sagittal_loss
            
            if idx % 5 == 0:
                logger(f"Epoch {epoch}, Batch {idx}/{len(train_loader)}, " 
                      f"Axial: {axial_loss:.3f}, Coronal: {coronal_loss:.3f}, Sagittal: {sagittal_loss:.3f}")
        
        # Average losses
        axial_train_loss /= len(train_loader)
        coronal_train_loss /= len(train_loader)
        sagittal_train_loss /= len(train_loader)
        train_loss = (axial_train_loss + coronal_train_loss + sagittal_train_loss) / 3
        
        epoch_time = time.time() - epoch_start_time
        logger(f"Epoch {epoch} completed in {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.3f} (Axial: {axial_train_loss:.3f}, "
              f"Coronal: {coronal_train_loss:.3f}, Sagittal: {sagittal_train_loss:.3f})")
        
        # Validation phase
        if epoch % args.val_interval == 0:
            val_start_time = time.time()
            model.eval()
            axial_val_loss = 0
            coronal_val_loss = 0
            sagittal_val_loss = 0
            
            for batch in val_loader:
                volume_x = batch["sCT"].to(device)
                volume_y = batch["CT"].to(device)
                
                axial_loss, coronal_loss, sagittal_loss = train_or_eval(
                    "val", model, volume_x, volume_y, optimizer, loss_fn, LOSS_FACTOR
                )
                
                axial_val_loss += axial_loss
                coronal_val_loss += coronal_loss
                sagittal_val_loss += sagittal_loss
            
            # Average losses
            axial_val_loss /= len(val_loader)
            coronal_val_loss /= len(val_loader)
            sagittal_val_loss /= len(val_loader)
            val_loss = (axial_val_loss + coronal_val_loss + sagittal_val_loss) / 3
            
            val_time = time.time() - val_start_time
            logger(f"Validation completed in {val_time:.2f}s - "
                  f"Validation Loss: {val_loss:.3f} (Axial: {axial_val_loss:.3f}, "
                  f"Coronal: {coronal_val_loss:.3f}, Sagittal: {sagittal_val_loss:.3f})")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(root_folder, f"best_model_fold_{cross_validation}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger(f"New best model saved at epoch {epoch} with validation loss: {val_loss:.3f}")
                
                # Test the best model
                test_start_time = time.time()
                axial_test_loss = 0
                coronal_test_loss = 0
                sagittal_test_loss = 0
                
                for batch in test_loader:
                    volume_x = batch["sCT"].to(device)
                    volume_y = batch["CT"].to(device)
                    
                    axial_loss, coronal_loss, sagittal_loss = train_or_eval(
                        "val", model, volume_x, volume_y, optimizer, loss_fn, LOSS_FACTOR
                    )
                    
                    axial_test_loss += axial_loss
                    coronal_test_loss += coronal_loss
                    sagittal_test_loss += sagittal_loss
                
                # Average losses
                axial_test_loss /= len(test_loader)
                coronal_test_loss /= len(test_loader)
                sagittal_test_loss /= len(test_loader)
                test_loss = (axial_test_loss + coronal_test_loss + sagittal_test_loss) / 3
                
                test_time = time.time() - test_start_time
                logger(f"Test evaluation completed in {test_time:.2f}s - "
                      f"Test Loss: {test_loss:.3f} (Axial: {axial_test_loss:.3f}, "
                      f"Coronal: {coronal_test_loss:.3f}, Sagittal: {sagittal_test_loss:.3f})")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(root_folder, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger(f"Best validation loss: {best_val_loss:.3f}")
    logger(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    # For examining data only:
    # examine_data()
    
    # For training:
    train()
