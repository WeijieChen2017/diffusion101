import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import json
import random

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
)

import time
from monai.data import CacheDataset, DataLoader
from global_config import global_config, set_param, get_param
# take the psnr as the metric from skimage
from skimage.metrics import peak_signal_noise_ratio as psnr

def printlog(message):
    log_txt_path = get_param("log_txt_path")
    # attach the current time as YYYY-MM-DD HH:MM:SS 
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    message = f"{current_time} {message}"
    print(message)
    with open(log_txt_path, "a") as f:
        f.write(message)
        f.write("\n")


@torch.inference_mode()
def test_diffusion_model_and_save(val_loader, model, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    print("Starting testing...")

    for idx_case, batch in enumerate(val_loader):
        printlog(f"Processing case {idx_case + 1}/{len(val_loader)}")

        pet = batch["PET"].to(device)  # Shape: (1, z, 256, 256)
        ct = batch["CT"].to(device)  # Ground truth CT, Shape: (1, z, 256, 256)
        len_z = pet.shape[1]  # Number of slices along the z-axis

        pred_ct_case = torch.zeros_like(ct)  # Placeholder for predicted CT, same shape as input CT

        # Generate predictions slice by slice
        for z in range(1, len_z - 1):
            # Create batch for model input
            cond = torch.zeros((1, 3, pet.shape[2], pet.shape[3])).to(device)  # Shape: (batch_size, 3, h, w)
            cond[0, 0, :, :] = pet[:, z - 1, :, :]
            cond[0, 1, :, :] = pet[:, z, :, :]
            cond[0, 2, :, :] = pet[:, z + 1, :, :]

            # Generate prediction using the diffusion model
            pred_slice = model.sample(batch_size=1, cond=cond).squeeze(0)  # Shape: (3, h, w)

            # Assign the middle slice prediction to the corresponding z position
            pred_ct_case[:, z, :, :] = pred_slice[1, :, :]  # Middle slice corresponds to current z

        # Clip predictions to [-1, 1], normalize to [0, 1], and ground truth is already in [0, 1]
        pred_ct_case = torch.clamp(pred_ct_case, min=-1, max=1)
        pred_ct_case = (pred_ct_case + 1) / 2.0  # Normalize to [0, 1]
        # ct = (ct + 1) / 2.0  # Normalize CT to [0, 1] as well

        # Compute MAE loss with a factor of 4000
        mae_loss = F.l1_loss(pred_ct_case, ct, reduction="mean") * 4000
        printlog(f"Case {idx_case + 1}: MAE Loss = {mae_loss.item():.6f}")

        # Save PET, CT, and predicted CT for this case
        case_data = {
            "PET": pet.cpu().numpy(),
            "CT": ct.cpu().numpy(),
            "Pred_CT": pred_ct_case.cpu().numpy(),
            "MAE": mae_loss.item()
        }
        save_path = os.path.join(output_dir, f"case_{idx_case + 1}.npz")
        np.savez_compressed(save_path, **case_data)

        printlog(f"Saved results for case {idx_case + 1} to {save_path}")

    printlog("Testing and saving completed.")



def train_or_eval_or_test_the_batch_cond(
        batch, 
        batch_size, 
        stage, model, 
        optimizer=None, 
        device=None, 
        # inceptionV3=None
    ):

    # if stage == "eval" or stage == "test":
    #     sampler = DDIMSampler(model)
    #     steps = get_param("steps")
    pet = batch["PET"] # 1, z, 256, 256
    ct = batch["CT"] # 1, z, 256, 256
    body = batch["BODY"]
    body = body > 0
    len_z = ct.shape[1]
    batch_per_eval = get_param("train_param")["batch_per_eval"]
    num_frames = get_param("num_frames")
    root_dir = get_param("root")
    # 256 to 128

    # body is the body mask, only masked region should be from 0-1 to -1 to 1
    # ct[body] = ct[body] * 2 - 1

    # ct = ct * 2 - 1
    # currently is simple from 0 to 1
    # ct = ct[:, :, 96:-96, 96:-96]

    # 1, z, 256, 256 tensor
    case_loss_first = 0.0
    case_loss_second = 0.0
    case_loss_third = 0.0

    indices_list_first = [i for i in range(1, ct.shape[1]-1)]

    random.shuffle(indices_list_first)

    # enumreate first dimension
    batch_size_count = 0
    batch_x = torch.zeros((batch_size, 3, ct.shape[2], ct.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[2], ct.shape[3]))
    for index in indices_list_first:
        batch_x[batch_size_count, 0, :, :] = pet[:, index-1, :, :]
        batch_x[batch_size_count, 1, :, :] = pet[:, index, :, :]
        batch_x[batch_size_count, 2, :, :] = pet[:, index+1, :, :]


        batch_y[batch_size_count, 0, :, :] = ct[:, index-1, :, :]
        batch_y[batch_size_count, 1, :, :] = ct[:, index, :, :]
        batch_y[batch_size_count, 2, :, :] = ct[:, index+1, :, :]
        # batch_y[batch_size_count, 0, 1, :, :] = ct[:, index-4, :, :]
        # batch_y[batch_size_count, 1, 1, :, :] = ct[:, index-3, :, :]
        # batch_y[batch_size_count, 2, 1, :, :] = ct[:, index-2, :, :]
        # batch_y[batch_size_count, 0, 2, :, :] = ct[:, index-1, :, :]
        # batch_y[batch_size_count, 1, 2, :, :] = ct[:, index, :, :]
        # batch_y[batch_size_count, 2, 2, :, :] = ct[:, index+1, :, :]
        # batch_y[batch_size_count, 0, 3, :, :] = ct[:, index+2, :, :]
        # batch_y[batch_size_count, 1, 3, :, :] = ct[:, index+3, :, :]
        # batch_y[batch_size_count, 2, 3, :, :] = ct[:, index+4, :, :]
        # batch_y[batch_size_count, 0, 4, :, :] = ct[:, index+5, :, :]
        # batch_y[batch_size_count, 1, 4, :, :] = ct[:, index+6, :, :]
        # batch_y[batch_size_count, 2, 4, :, :] = ct[:, index+7, :, :]

        batch_size_count += 1

        if batch_size_count < batch_size and index != indices_list_first[-1]:
            continue
        else:
            # # we get a batch
            # save_batch_y = batch_y.cpu().numpy()
            # save_name = f"{root_dir}/batch_y_{index}_masked.npy"
            # np.save(save_name, save_batch_y)
            # printlog(f"save batch_y to {save_name}")
            # exit()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if stage == "train":
                optimizer.zero_grad()
                loss = model(
                    img=batch_y,
                    cond=batch_x,
                )
                loss.backward()
                optimizer.step()
                case_loss_first += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    loss = model(
                        img=batch_y,
                        cond=batch_x,
                    )
                    case_loss_first += loss.item()

            batch_size_count = 0
        
    case_loss_first = case_loss_first / (len(indices_list_first) // batch_size + 1)
        # if (stage == "eval" or stage == "test") and batch_per_eval > 0:
        #     case_loss_first = case_loss_first / batch_eval_count
        # else:
        #     case_loss_first = case_loss_first / (len(indices_list_first) // batch_size + 1)
    
    # # enumreate second dimension
    # batch_size_count = 0
    # batch_eval_count = 0
    # batch_y = torch.zeros((batch_size, , ct.shape[1], ct.shape[3]))

    # for indices in indices_list_second:
    #     slice_y = ct[:, :, indices-1:indices+2, :]
    #     # adjust the index order
    #     slice_y = slice_y.permute(0, 2, 1, 3)
    #     batch_size_count += 1

    #     batch_y[batch_size_count-1] = slice_y

    #     if batch_size_count < batch_size and indices != indices_list_second[-1]:
    #         continue
    #     else:
    #         # we get a batch
    #         batch_y = batch_y.to(device)
    #         encoded_batch_y = model.first_stage_model.encode(batch_y)
    #         # print(batch_x.size(), batch_y.size())
    #         if stage == "train":
    #             optimizer.zero_grad()
    #             loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
    #             loss.backward()
    #             optimizer.step()
    #             case_loss_second += loss.item()
    #         elif stage == "eval" or stage == "test":
    #             with torch.no_grad():
    #                 # loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
    #                 # case_loss_first += loss.item()
    #                 encoded_batch_x = model.first_stage_model.encode(batch_x) / vqs
    #                 shape = (encoded_batch_x.shape[1],)+encoded_batch_x.shape[2:]
    #                 samples_ddim, _ = sampler.sample(
    #                     S=steps,
    #                     conditioning=encoded_batch_x,
    #                     batch_size=encoded_batch_x.shape[0],
    #                     shape=shape,
    #                     verbose=False
    #                 )
    #                 recon_batch_y = model.decode_first_stage(samples_ddim)
    #                 recon_batch_y = recon_batch_y.cpu().numpy().transpose(0,2,3,1)
    #                 recon_batch_y = np.clip((recon_batch_y+1.0)/2.0, 0.0, 1.0)
    #                 true_y = batch_y.cpu().numpy().transpose(0,2,3,1)
    #                 true_y = np.clip((true_y+1.0)/2.0, 0.0, 1.0)
    #                 # calculate the L1 loss
    #                 loss = np.mean(np.abs(recon_batch_y-true_y))
    #                 case_loss_second += loss
    #                 batch_eval_count += 1
    #                 if batch_eval_count == batch_per_eval:
    #                     # stop the for loop
    #                     break
    #         batch_size_count = 0
        
    #     if (stage == "eval" or stage == "test") and batch_per_eval > 0:
    #         case_loss_second = case_loss_second / batch_eval_count
    #     else:
    #         case_loss_second = case_loss_second / (len(indices_list_second) // batch_size + 1)
    
    # # enumreate third dimension
    # batch_size_count = 0
    # batch_eval_count = 0
    # batch_x = torch.zeros((batch_size, 3, pet.shape[1], pet.shape[2]))
    # batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[2]))

    # for indices in indices_list_third:
    #     slice_x = pet[:, :, :, indices-1:indices+2]
    #     slice_y = ct[:, :, :, indices-1:indices+2]
    #     # adjust the index order
    #     slice_x = slice_x.permute(0, 3, 1, 2)
    #     slice_y = slice_y.permute(0, 3, 1, 2)
    #     batch_size_count += 1

    #     batch_x[batch_size_count-1] = slice_x
    #     batch_y[batch_size_count-1] = slice_y

    #     if batch_size_count < batch_size and indices != indices_list_third[-1]:
    #         continue
    #     else:
    #         # we get a batch
    #         batch_x = batch_x.to(device)
    #         batch_y = batch_y.to(device)
    #         encoded_batch_y = model.first_stage_model.encode(batch_y)
    #         if stage == "train":
    #             optimizer.zero_grad()
    #             loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
    #             loss.backward()
    #             optimizer.step()
    #             case_loss_third += loss.item()
    #         elif stage == "eval" or stage == "test":
    #             with torch.no_grad():
    #                 # loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
    #                 # case_loss_first += loss.item()
    #                 encoded_batch_x = model.first_stage_model.encode(batch_x) / vqs
    #                 shape = (encoded_batch_x.shape[1],)+encoded_batch_x.shape[2:]
    #                 samples_ddim, _ = sampler.sample(
    #                     S=steps,
    #                     conditioning=encoded_batch_x,
    #                     batch_size=encoded_batch_x.shape[0],
    #                     shape=shape,
    #                     verbose=False
    #                 )
    #                 recon_batch_y = model.decode_first_stage(samples_ddim)
    #                 recon_batch_y = recon_batch_y.cpu().numpy().transpose(0,2,3,1)
    #                 recon_batch_y = np.clip((recon_batch_y+1.0)/2.0, 0.0, 1.0)
    #                 true_y = batch_y.cpu().numpy().transpose(0,2,3,1)
    #                 true_y = np.clip((true_y+1.0)/2.0, 0.0, 1.0)
    #                 # calculate the L1 loss
    #                 loss = np.mean(np.abs(recon_batch_y-true_y))
    #                 case_loss_third += loss
    #                 batch_eval_count += 1
    #                 if batch_eval_count == batch_per_eval:
    #                     # stop the for loop
    #                     break
    #         batch_size_count = 0
        
    #     if (stage == "eval" or stage == "test") and batch_per_eval > 0:
    #         case_loss_third = case_loss_third / batch_eval_count
    #     else:
    #         case_loss_third = case_loss_third / (len(indices_list_third) // batch_size + 1)

    return case_loss_first, case_loss_second, case_loss_third


# def train_or_eval_or_test_the_batch_cond(batch, batch_size, stage, model, optimizer=None, device=None):

    # if stage == "eval" or stage == "test":
    #     sampler = DDIMSampler(model)
    #     steps = get_param("steps")

    pet = batch["PET"] # 1, z, 256, 256
    ct = batch["CT"] # 1, z, 256, 256
    len_z = pet.shape[1]

    es = get_param("es")
    vqs = get_param("vq_scaling")
    batch_per_eval = get_param("train_param")["batch_per_eval"]
    zac = get_param("z_as_channel")
    eo = (zac - 1) // 2 # ends_offset

    pet = pet * 2 - 1
    ct = ct * 2 - 1

    # if pet size and ct size are not the same skip this batch
    if pet.shape != ct.shape:
        printlog(f"skip this batch, pet shape: {pet.shape}, ct shape: {ct.shape}")
        return 1.0, 1.0, 1.0

    # 1, z, 256, 256 tensor
    case_loss_first = 0.0
    case_loss_second = 0.0
    case_loss_third = 0.0

    # pad shape
    if len_z % es != 0:
        pad_size = es - len_z % es
        pet = torch.nn.functional.pad(pet, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
        ct = torch.nn.functional.pad(ct, (0, 0, 0, 0, 0, pad_size, 0, 0), mode='constant', value=0)

    # printlog(f"pet shape {pet.shape}, ct shape {ct.shape}")

    indices_list_first = [i for i in range(1, pet.shape[1]-eo)]
    indices_list_second = [i for i in range(1, pet.shape[2]-eo)]
    indices_list_third = [i for i in range(1, pet.shape[3]-eo)]

    random.shuffle(indices_list_first)
    random.shuffle(indices_list_second)
    random.shuffle(indices_list_third)

    # enumreate first dimension
    batch_size_count = 0
    batch_eval_count = 0
    batch_x = torch.zeros((batch_size, 1, zac, pet.shape[2], pet.shape[3]))
    batch_y = torch.zeros((batch_size, 1, zac, ct.shape[2], ct.shape[3]))
    for indices in indices_list_first:
        slice_x = pet[:, indices-eo:indices+eo+1, :, :]
        slice_y = ct[:, indices-eo:indices+eo+1, :, :]
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_first[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if stage == "train":
                encoded_batch_y = model.first_stage_model.encode(batch_y)
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_first += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    # loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    # case_loss_first += loss.item()
                    encoded_batch_x = model.first_stage_model.encode(batch_x) / vqs
                    shape = (encoded_batch_x.shape[1],)+encoded_batch_x.shape[2:]
                    samples_ddim, _ = sampler.sample(
                        S=steps,
                        conditioning=encoded_batch_x,
                        batch_size=encoded_batch_x.shape[0],
                        shape=shape,
                        verbose=False
                    )
                    recon_batch_y = model.decode_first_stage(samples_ddim)
                    recon_batch_y = recon_batch_y.cpu().numpy().transpose(0,2,3,1)
                    recon_batch_y = np.clip((recon_batch_y+1.0)/2.0, 0.0, 1.0)
                    true_y = batch_y.cpu().numpy().transpose(0,2,3,1)
                    true_y = np.clip((true_y+1.0)/2.0, 0.0, 1.0)
                    # calculate the L1 loss
                    loss = np.mean(np.abs(recon_batch_y-true_y))
                    case_loss_first += loss
                    batch_eval_count += 1
                    if batch_eval_count == batch_per_eval:
                        # stop the for loop
                        break

            batch_size_count = 0
        
        if (stage == "eval" or stage == "test") and batch_per_eval > 0:
            case_loss_first = case_loss_first / batch_eval_count
        else:
            case_loss_first = case_loss_first / (len(indices_list_first) // batch_size + 1)
    
    # enumreate second dimension
    batch_size_count = 0
    batch_eval_count = 0
    batch_x = torch.zeros((batch_size, 3, pet.shape[1], pet.shape[3]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[3]))

    for indices in indices_list_second:
        slice_x = pet[:, :, indices-1:indices+2, :]
        slice_y = ct[:, :, indices-1:indices+2, :]
        # adjust the index order
        slice_x = slice_x.permute(0, 2, 1, 3)
        slice_y = slice_y.permute(0, 2, 1, 3)
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_second[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded_batch_y = model.first_stage_model.encode(batch_y)
            # print(batch_x.size(), batch_y.size())
            if stage == "train":
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_second += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    # loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    # case_loss_first += loss.item()
                    encoded_batch_x = model.first_stage_model.encode(batch_x) / vqs
                    shape = (encoded_batch_x.shape[1],)+encoded_batch_x.shape[2:]
                    samples_ddim, _ = sampler.sample(
                        S=steps,
                        conditioning=encoded_batch_x,
                        batch_size=encoded_batch_x.shape[0],
                        shape=shape,
                        verbose=False
                    )
                    recon_batch_y = model.decode_first_stage(samples_ddim)
                    recon_batch_y = recon_batch_y.cpu().numpy().transpose(0,2,3,1)
                    recon_batch_y = np.clip((recon_batch_y+1.0)/2.0, 0.0, 1.0)
                    true_y = batch_y.cpu().numpy().transpose(0,2,3,1)
                    true_y = np.clip((true_y+1.0)/2.0, 0.0, 1.0)
                    # calculate the L1 loss
                    loss = np.mean(np.abs(recon_batch_y-true_y))
                    case_loss_second += loss
                    batch_eval_count += 1
                    if batch_eval_count == batch_per_eval:
                        # stop the for loop
                        break
            batch_size_count = 0
        
        if (stage == "eval" or stage == "test") and batch_per_eval > 0:
            case_loss_second = case_loss_second / batch_eval_count
        else:
            case_loss_second = case_loss_second / (len(indices_list_second) // batch_size + 1)
    
    # enumreate third dimension
    batch_size_count = 0
    batch_eval_count = 0
    batch_x = torch.zeros((batch_size, 3, pet.shape[1], pet.shape[2]))
    batch_y = torch.zeros((batch_size, 3, ct.shape[1], ct.shape[2]))

    for indices in indices_list_third:
        slice_x = pet[:, :, :, indices-1:indices+2]
        slice_y = ct[:, :, :, indices-1:indices+2]
        # adjust the index order
        slice_x = slice_x.permute(0, 3, 1, 2)
        slice_y = slice_y.permute(0, 3, 1, 2)
        batch_size_count += 1

        batch_x[batch_size_count-1] = slice_x
        batch_y[batch_size_count-1] = slice_y

        if batch_size_count < batch_size and indices != indices_list_third[-1]:
            continue
        else:
            # we get a batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded_batch_y = model.first_stage_model.encode(batch_y)
            if stage == "train":
                optimizer.zero_grad()
                loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                loss.backward()
                optimizer.step()
                case_loss_third += loss.item()
            elif stage == "eval" or stage == "test":
                with torch.no_grad():
                    # loss, loss_dict = model(x=encoded_batch_y, c=batch_x)
                    # case_loss_first += loss.item()
                    encoded_batch_x = model.first_stage_model.encode(batch_x) / vqs
                    shape = (encoded_batch_x.shape[1],)+encoded_batch_x.shape[2:]
                    samples_ddim, _ = sampler.sample(
                        S=steps,
                        conditioning=encoded_batch_x,
                        batch_size=encoded_batch_x.shape[0],
                        shape=shape,
                        verbose=False
                    )
                    recon_batch_y = model.decode_first_stage(samples_ddim)
                    recon_batch_y = recon_batch_y.cpu().numpy().transpose(0,2,3,1)
                    recon_batch_y = np.clip((recon_batch_y+1.0)/2.0, 0.0, 1.0)
                    true_y = batch_y.cpu().numpy().transpose(0,2,3,1)
                    true_y = np.clip((true_y+1.0)/2.0, 0.0, 1.0)
                    # calculate the L1 loss
                    loss = np.mean(np.abs(recon_batch_y-true_y))
                    case_loss_third += loss
                    batch_eval_count += 1
                    if batch_eval_count == batch_per_eval:
                        # stop the for loop
                        break
            batch_size_count = 0
        
        if (stage == "eval" or stage == "test") and batch_per_eval > 0:
            case_loss_third = case_loss_third / batch_eval_count
        else:
            case_loss_third = case_loss_third / (len(indices_list_third) // batch_size + 1)

    return case_loss_first, case_loss_second, case_loss_third

def prepare_dataset(data_div, invlove_test=False):
    
    cv = get_param("cv")
    root = get_param("root")
    
    # cv = 0, 1, 2, 3, 4
    cv_test = cv
    cv_val = (cv+1)%5
    cv_train = [(cv+2)%5, (cv+3)%5, (cv+4)%5]

    train_list = data_div[f"cv{cv_train[0]}"] + data_div[f"cv{cv_train[1]}"] + data_div[f"cv{cv_train[2]}"]
    val_list = data_div[f"cv{cv_val}"]
    test_list = data_div[f"cv{cv_test}"]

    set_param("train_list", train_list)
    set_param("val_list", val_list)
    set_param("test_list", test_list)

    print(f"train_list:", train_list)
    print(f"val_list:", val_list)
    print(f"test_list:", test_list)

    # train_list: ['E4058', 'E4217', 'E4166', 'E4165', 'E4092', 'E4163', 'E4193', 'E4105', 'E4125', 'E4198', 'E4157', 'E4139', 'E4207', 'E4106', 'E4068', 'E4241', 'E4219', 'E4078', 'E4147', 'E4138', 'E4096', 'E4152', 'E4073', 'E4181', 'E4187', 'E4099', 'E4077', 'E4134', 'E4091', 'E4144', 'E4114', 'E4130', 'E4103', 'E4239', 'E4183', 'E4208', 'E4120', 'E4220', 'E4137', 'E4069', 'E4189', 'E4182']
    # val_list: ['E4216', 'E4081', 'E4118', 'E4074', 'E4079', 'E4094', 'E4115', 'E4237', 'E4084', 'E4061', 'E4055', 'E4098', 'E4232']
    # test_list: ['E4128', 'E4172', 'E4238', 'E4158', 'E4129', 'E4155', 'E4143', 'E4197', 'E4185', 'E4131', 'E4162', 'E4066', 'E4124']

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })

    for hashname in val_list:
        val_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })

    for hashname in test_list:
        test_path_list.append({
            "PET": f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })

    # save the data division file
    data_division_file = os.path.join(root, "data_division.json")
    data_division_dict = {
        "train": train_path_list,
        "val": val_path_list,
        "test": test_path_list,
    }
    for key in data_division_dict.keys():
        print(key)
        for key2 in data_division_dict[key]:
            print(key2)

    with open(data_division_file, "w") as f:
        json.dump(data_division_dict, f, indent=4)

    input_modality = ["PET", "CT", "BODY"]  

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
            # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["x"], 
            #     roi_size=(img_size, img_size, in_channel), 
            #     random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),

        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=input_modality, image_only=True),
            EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
            # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["x"], 
            #     roi_size=(img_size, img_size, in_channel), 
            #     random_size=False),
            # RandSpatialCropd(
            #     keys=input_modality_dict["y"],
            #     roi_size=(img_size, img_size, out_channel),
            #     random_size=False),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["x"],
            #     channel_dim=-1),
            # EnsureChannelFirstd(
            #     keys=input_modality_dict["y"],
            #     channel_dim="none" if out_channel == 1 else -1),
        ]
    )
    if invlove_test:
        test_transforms = Compose(
            [
                LoadImaged(keys=input_modality, image_only=True),
                EnsureChannelFirstd(keys=input_modality, channel_dim=-1),
                # NormalizeIntensityd(keys=input_modality, nonzero=True, channel_wise=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["x"], 
                #     roi_size=(img_size, img_size, in_channel), 
                #     random_size=False),
                # RandSpatialCropd(
                #     keys=input_modality_dict["y"],
                #     roi_size=(img_size, img_size, out_channel),
                #     random_size=False),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["x"],
                #     channel_dim=-1),
                # EnsureChannelFirstd(
                #     keys=input_modality_dict["y"],
                #     channel_dim="none" if out_channel == 1 else -1),
            ]
        )

    

    train_ds = CacheDataset(
        data=train_path_list,
        transform=train_transforms,
        # cache_num=num_train_files,
        cache_rate=get_param("data_param")["dataset"]["train"]["cache_rate"],
        num_workers=get_param("data_param")["dataset"]["train"]["num_workers"],
    )

    val_ds = CacheDataset(
        data=val_path_list,
        transform=val_transforms, 
        # cache_num=num_val_files,
        cache_rate=get_param("data_param")["dataset"]["val"]["cache_rate"],
        num_workers=get_param("data_param")["dataset"]["val"]["num_workers"],
    )
    if invlove_test:
        test_ds = CacheDataset(
            data=test_path_list,
            transform=test_transforms,
            # cache_num=num_test_files,
            cache_rate=get_param("data_param")["dataset"]["test"]["cache_rate"],
            num_workers=get_param("data_param")["dataset"]["test"]["num_workers"],
        )

    train_loader = DataLoader(
        train_ds, 
        batch_size=1,
        shuffle=True,
        num_workers=get_param("data_param")["dataloader"]["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1,
        shuffle=False,
        num_workers=get_param("data_param")["dataloader"]["val"]["num_workers"],
    )

    if invlove_test:
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=get_param("data_param")["dataloader"]["test"]["num_workers"],
        )

    if not invlove_test:
        test_loader = None
    
    return train_loader, val_loader, test_loader

# from torchvision.models import inception_v3
# from scipy.linalg import sqrtm
# from torchvision import transforms
# import torch.nn.functional as F

# def load_inception_model(weights_path):
#     """
#     Load InceptionV3 model from a given path. If the model weights are not found at the path,
#     download the pretrained model, save the weights to the path, and return the model.
    
#     Args:
#         weights_path (str): Path to the file where pretrained weights are stored or will be saved.
    
#     Returns:
#         torch.nn.Module: The InceptionV3 model loaded with weights.
#     """
#     model = inception_v3(pretrained=False, transform_input=False)  # Initialize the model

#     if os.path.exists(weights_path):
#         # Load weights locally if the file exists
#         print(f"Loading pretrained weights from {weights_path}...")
#         model.load_state_dict(torch.load(weights_path))
#     else:
#         # Download pretrained weights, save them locally, and load them
#         print(f"Pretrained weights not found at {weights_path}. Downloading and saving locally...")
#         model = inception_v3(pretrained=True, transform_input=False)
#         torch.save(model.state_dict(), weights_path)
    
#     model.eval()  # Set model to evaluation mode
#     return model

# def get_features(images, model):
#     with torch.no_grad():
#         features = model(images).detach()
#     return features

# def compute_FID(real_y, recon_y, inception):
#     # Load pre-trained InceptionV3 model
#     # inception = inception_v3(pretrained=True, transform_input=False)
#     # inception.eval()  # Set the model to evaluation mode
    
#     # Step 1: Rescale to [0, 1]
#     real_y = (real_y + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
#     recon_y = (recon_y + 1) / 2.0 # Rescale from [-1, 1] to [0, 1]
    
#     # Step 3: Resize to 299x299
#     real_y = F.interpolate(real_y, size=(299, 299), mode='bilinear', align_corners=False)
#     recon_y = F.interpolate(recon_y, size=(299, 299), mode='bilinear', align_corners=False)
    
#     # Step 4: Normalize using ImageNet statistics
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
#     real_y = torch.stack([normalize(img) for img in real_y])
#     recon_y = torch.stack([normalize(img) for img in recon_y])

#     # Extract features for both real and reconstructed images
#     real_features = get_features(real_y, inception)
#     recon_features = get_features(recon_y, inception)

#     real_features_np = real_features.cpu().numpy()
#     recon_features_np = recon_features.cpu().numpy()

#     # Compute mean and covariance
#     real_mu = np.mean(real_features_np, axis=0)
#     real_sigma = np.cov(real_features_np, rowvar=False)

#     recon_mu = np.mean(recon_features_np, axis=0)
#     recon_sigma = np.cov(recon_features_np, rowvar=False)

#     # Compute the squared difference between means
#     mean_diff = np.sum((real_mu - recon_mu) ** 2)

#     # Compute the square root of the product of covariance matrices
#     cov_sqrt = sqrtm(real_sigma @ recon_sigma)

#     # Handle numerical errors from sqrtm (e.g., small imaginary components)
#     if np.iscomplexobj(cov_sqrt):
#         cov_sqrt = cov_sqrt.real

#     # Compute the FID score
#     fid = mean_diff + np.trace(real_sigma + recon_sigma - 2 * cov_sqrt)

#     return fid