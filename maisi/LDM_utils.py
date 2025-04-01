import os
import time
import json
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import random
# import pytorch_lightning as pl

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
)

from monai.data import CacheDataset, DataLoader


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers_loader: int
    num_workers_cache: int
    cache_rate: float

@dataclass
class DataConfig:
    root_folder: str
    cross_validation: int
    input_modality: List[str]
    train: DataLoaderConfig
    val: DataLoaderConfig
    test: DataLoaderConfig
    
    @classmethod
    def from_dict(cls, config_dict):
        train_loader = DataLoaderConfig(
            batch_size=config_dict["data_loader_params"]["train"]["batch_size"],
            shuffle=config_dict["data_loader_params"]["train"]["shuffle"],
            num_workers_loader=config_dict["data_loader_params"]["train"]["num_workers_loader"],
            num_workers_cache=config_dict["data_loader_params"]["train"]["num_workers_cache"],
            cache_rate=config_dict["data_loader_params"]["train"]["cache_rate"]
        )
        
        val_loader = DataLoaderConfig(
            batch_size=config_dict["data_loader_params"]["val"]["batch_size"],
            shuffle=config_dict["data_loader_params"]["val"]["shuffle"],
            num_workers_loader=config_dict["data_loader_params"]["val"]["num_workers_loader"],
            num_workers_cache=config_dict["data_loader_params"]["val"]["num_workers_cache"],
            cache_rate=config_dict["data_loader_params"]["val"]["cache_rate"]
        )
        
        test_loader = DataLoaderConfig(
            batch_size=config_dict["data_loader_params"]["test"]["batch_size"],
            shuffle=config_dict["data_loader_params"]["test"]["shuffle"],
            num_workers_loader=config_dict["data_loader_params"]["test"]["num_workers_loader"],
            num_workers_cache=config_dict["data_loader_params"]["test"]["num_workers_cache"],
            cache_rate=config_dict["data_loader_params"]["test"]["cache_rate"]
        )
        
        return cls(
            root_folder=config_dict["root_folder"],
            cross_validation=config_dict["cross_validation"],
            input_modality=config_dict["model_step1_params"]["input_modality"],
            train=train_loader,
            val=val_loader,
            test=test_loader
        )


def prepare_dataset(data_div_json, config: DataConfig):
    
    with open(data_div_json, "r") as f:
        data_div = json.load(f)
    
    cv = config.cross_validation

    train_list = data_div[cv]["train"]
    val_list = data_div[cv]["val"]
    test_list = data_div[cv]["test"]

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for casename in train_list:
        train_path_list.append({
            "CT": f"LDM_adapter/data/CT/CTAC_{casename}_cropped.nii.gz",
            "sCT": f"LDM_adapter/data/sCT/CTAC_{casename}_TS_MAISI.nii.gz",
        })

    for casename in val_list:
        val_path_list.append({
            "CT": f"LDM_adapter/data/CT/CTAC_{casename}_cropped.nii.gz",
            "sCT": f"LDM_adapter/data/sCT/CTAC_{casename}_TS_MAISI.nii.gz",
        })

    for casename in test_list:
        test_path_list.append({
            "CT": f"LDM_adapter/data/CT/CTAC_{casename}_cropped.nii.gz",
            "sCT": f"LDM_adapter/data/sCT/CTAC_{casename}_TS_MAISI.nii.gz",
        })

    # save the data division file
    data_division_file = os.path.join(config.root_folder, "data_division.json")
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

    # set the data transform
    train_transforms = Compose(
        [
            LoadImaged(keys=config.input_modality, image_only=True),
            EnsureChannelFirstd(keys=config.input_modality, channel_dim=-1),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=config.input_modality, image_only=True),
            EnsureChannelFirstd(keys=config.input_modality, channel_dim=-1),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=config.input_modality, image_only=True),
            EnsureChannelFirstd(keys=config.input_modality, channel_dim=-1),
        ]
    )

    train_ds = CacheDataset(
        data=train_path_list,
        transform=train_transforms,
        cache_rate=config.train.cache_rate,
        num_workers=config.train.num_workers_cache,
    )

    val_ds = CacheDataset(
        data=val_path_list,
        transform=val_transforms, 
        cache_rate=config.val.cache_rate,
        num_workers=config.val.num_workers_cache,
    )

    test_ds = CacheDataset(
        data=test_path_list,
        transform=test_transforms,
        cache_rate=config.test.cache_rate,
        num_workers=config.test.num_workers_cache,
    )

    train_loader = DataLoader(train_ds, 
                            batch_size=config.train.batch_size,
                            shuffle=config.train.shuffle,
                            num_workers=config.train.num_workers_loader,
    )
    
    val_loader = DataLoader(val_ds, 
                            batch_size=config.val.batch_size,
                            shuffle=config.val.shuffle,
                            num_workers=config.val.num_workers_loader,
    )

    test_loader = DataLoader(test_ds,
                            batch_size=config.test.batch_size,
                            shuffle=config.test.shuffle,
                            num_workers=config.test.num_workers_loader,
    )

    return train_loader, val_loader, test_loader


class simple_logger():
    def __init__(self, log_file_path, global_config):
        self.log_file_path = log_file_path
        self.log_dict = dict()
        self.IS_LOGGER_WANDB = global_config["IS_LOGGER_WANDB"]
        if self.IS_LOGGER_WANDB:
            self.wandb_run = global_config["wandb_run"]
    
    def log(self, global_epoch, key, msg):
        if key not in self.log_dict.keys():
            self.log_dict[key] = dict()
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dict[key] = {
            "time": current_time,
            "global_epoch": global_epoch,
            "msg": msg
        }
        log_str = f"{current_time} Global epoch: {global_epoch}, {key}, {msg}\n"
        with open(self.log_file_path, "a") as f:
            f.write(log_str)
        print(log_str)

        # log to wandb if msg is number
        if self.IS_LOGGER_WANDB and isinstance(msg, (int, float)):
            self.wandb_run.log({key: msg})




def two_segment_scale(arr, MIN, MID, MAX, MIQ):
    # Create an empty array to hold the scaled results
    scaled_arr = np.zeros_like(arr, dtype=np.float32)

    # First segment: where arr <= MID
    mask1 = arr <= MID
    scaled_arr[mask1] = (arr[mask1] - MIN) / (MID - MIN) * MIQ

    # Second segment: where arr > MID
    mask2 = arr > MID
    scaled_arr[mask2] = MIQ + (arr[mask2] - MID) / (MAX - MID) * (1 - MIQ)
    
    return scaled_arr


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                dropout, temb_channels=512):
      super().__init__()
      self.in_channels = in_channels
      out_channels = in_channels if out_channels is None else out_channels
      self.out_channels = out_channels
      self.use_conv_shortcut = conv_shortcut

      self.norm1 = Normalize(in_channels)
      self.conv1 = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
      if temb_channels > 0:
          self.temb_proj = torch.nn.Linear(temb_channels,
                                            out_channels)
      self.norm2 = Normalize(out_channels)
      self.dropout = torch.nn.Dropout(dropout)
      self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
      if self.in_channels != self.out_channels:
          if self.use_conv_shortcut:
              self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
          else:
              self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)

  def forward(self, x, temb):
      h = x
      h = self.norm1(h)
      h = nonlinearity(h)
      h = self.conv1(h)

      if temb is not None:
          h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

      h = self.norm2(h)
      h = nonlinearity(h)
      h = self.dropout(h)
      h = self.conv2(h)

      if self.in_channels != self.out_channels:
          if self.use_conv_shortcut:
              x = self.conv_shortcut(x)
          else:
              x = self.nin_shortcut(x)

      return x+h
  

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
        
def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
    

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)


        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, # double it for concat
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                #  ckpt_path=None,
                #  ignore_keys=[],
                #  image_key="image",
                 freeze_encoder=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        # self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.out_conv = nn.Conv2d(ddconfig["out_ch"], 1, 1)

        if freeze_encoder:
            self.freeze_model_part("encoder")
            self.freeze_model_part("quant_conv")
    
    def freeze_model_part(self, model_part):
        if model_part == "encoder":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif model_part == "decoder":
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif model_part == "quant_conv":
            for param in self.quant_conv.parameters():
                param.requires_grad = False
        elif model_part == "post_quant_conv":
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
        elif model_part == "out_conv":
            for param in self.out_conv.parameters():
                param.requires_grad = False
        else:
            print("Invalid model part to freeze.")
            return

    def init_random_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Initialized model weights randomly.")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant = self.encode(input)
        dec = self.decode(quant)
        out = self.out_conv(dec)
        return out
