cv = 0
root_dir = f"projects/v3_img_petCond_acs_cv{cv}/"
results_folder = root_dir+"test_results_ddpm_batch_128/"
model_pretrain_weights = "vq_f4.pth"
data_div_file = root_dir+"data_division.json"
test_case_idx = 1  # -1 means process all cases, 1 means only first case

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import json
import glob
import os
import torch.nn.functional as F
import pandas as pd

from train_v3_embedding_decode_utils import nnVQModel

# Set device explicitly to CUDA:0
device = torch.device("cuda:0")

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
    print(f"The model is successfully loaded from {model_pretrain_weights}")
else:
    print(f"Warning: Pretrained weights file '{model_pretrain_weights}' not found. Using randomly initialized weights.")

# Load data division file
with open(data_div_file, "r") as f:
    data_div = json.load(f)

# Get test cases from data division file
test_cases = []
for key in data_div.keys():
    if key == "test":  # Only look at test cases
        if isinstance(data_div[key], list):
            for path_dict in data_div[key]:
                if isinstance(path_dict, dict) and "filename" in path_dict:
                    test_cases.append(path_dict["filename"])

# If test_case_idx is specified, only process that case
if test_case_idx > 0 and test_case_idx <= len(test_cases):
    test_cases = [test_cases[test_case_idx - 1]]
    print(f"Processing only test case {test_case_idx}: {test_cases[0]}")
else:
    print(f"Processing all {len(test_cases)} test cases")

# Create a dictionary to store all slices for each case and view
case_data = {}

# Search for all related NPZ files in the results folder
for case_idx, case_name in enumerate(test_cases, 1):
    case_data[case_name] = {
        "axial": [],
        "coronal": [],
        "sagittal": []
    }
    
    # For each view
    for view in ["axial", "coronal", "sagittal"]:
        # Find all slice files for this case and view
        slice_pattern = f"{case_name}_case_{case_idx}_{view}_slice_*.npz"
        slice_files = glob.glob(os.path.join(results_folder, slice_pattern))
        
        # Sort files by slice number
        slice_files.sort(key=lambda x: int(x.split("_slice_")[-1].split(".")[0]))
        
        # Store the sorted file paths
        case_data[case_name][view] = slice_files

# Print summary
for case_name in case_data:
    print(f"\nCase: {case_name}")
    for view in ["axial", "coronal", "sagittal"]:
        num_slices = len(case_data[case_name][view])
        print(f"  {view}: {num_slices} slices")

# Create output directory for decoded results
decoded_results_folder = os.path.join(root_dir, "decoded_results_ddpm_batch_128")
os.makedirs(decoded_results_folder, exist_ok=True)

# Create lists to store results
results_data = []

# Process each case
with torch.no_grad():
    for case_name in case_data:
        print(f"\nProcessing case: {case_name}")
        for view in ["axial", "coronal", "sagittal"]:
            total_pred_loss = 0.0
            total_vq_pred_loss = 0.0
            num_slices = len(case_data[case_name][view])
            
            print(f"  View: {view}")
            
            for slice_path in case_data[case_name][view]:
                # Load NPZ file
                data = np.load(slice_path)
                
                # Convert numpy arrays to PyTorch tensors and add batch dimension
                gt_emb = torch.from_numpy(data['gt_embedding']).unsqueeze(0).to(device)
                pred_emb = torch.from_numpy(data['pred_embedding']).unsqueeze(0).to(device)
                vq_pred_emb = torch.from_numpy(data['vq_pred_embedding']).unsqueeze(0).to(device)
                
                # Denormalize by multiplying by 5
                gt_emb = gt_emb * 5
                pred_emb = pred_emb * 5
                vq_pred_emb = vq_pred_emb * 5
                
                # Decode embeddings
                gt_dec = nnmodel.decode(gt_emb)[:, 1:2]  # Take middle channel
                pred_dec = nnmodel.decode(pred_emb)[:, 1:2]
                vq_pred_dec = nnmodel.decode(vq_pred_emb)[:, 1:2]
                
                # Squeeze tensors for loss computation
                gt_dec = gt_dec.squeeze()
                pred_dec = pred_dec.squeeze()
                vq_pred_dec = vq_pred_dec.squeeze()
                
                # Compute L1 losses and scale to MAE (*4000)
                pred_loss = F.l1_loss(gt_dec, pred_dec) * 4000
                vq_pred_loss = F.l1_loss(gt_dec, vq_pred_dec) * 4000
                
                # Accumulate losses
                total_pred_loss += pred_loss.item()
                total_vq_pred_loss += vq_pred_loss.item()
                
                # Create output filename based on input filename
                base_filename = os.path.basename(slice_path)
                output_filename = os.path.join(decoded_results_folder, f"decoded_{base_filename}")
                
                # Save denormalized results to NPZ file
                np.savez(
                    output_filename,
                    gt_decoded=gt_dec.cpu().numpy(),
                    pred_decoded=pred_dec.cpu().numpy(),
                    vq_pred_decoded=vq_pred_dec.cpu().numpy(),
                    pred_mae=pred_loss.item(),
                    vq_pred_mae=vq_pred_loss.item()
                )
            
            # Calculate average losses for this view
            avg_pred_loss = total_pred_loss / num_slices
            avg_vq_pred_loss = total_vq_pred_loss / num_slices
            
            # Store results
            results_data.append({
                'Case': case_name,
                'View': view,
                'Num Slices': num_slices,
                'Pred MAE': avg_pred_loss,
                'VQ Pred MAE': avg_vq_pred_loss
            })
            
            print(f"    Average Pred MAE: {avg_pred_loss:.6f}")
            print(f"    Average VQ Pred MAE: {avg_vq_pred_loss:.6f}")
            print(f"    Total slices processed: {num_slices}")

# Create DataFrame and save to Excel
df = pd.DataFrame(results_data)

# Calculate overall statistics
overall_stats = pd.DataFrame({
    'Metric': ['Overall Average', 'Std Dev'],
    'Pred MAE': [df['Pred MAE'].mean(), df['Pred MAE'].std()],
    'VQ Pred MAE': [df['VQ Pred MAE'].mean(), df['VQ Pred MAE'].std()]
})

# Create Excel writer object
excel_path = os.path.join(root_dir, 'decode_results.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    # Write detailed results
    df.to_excel(writer, sheet_name='Detailed Results', index=False)
    
    # Write summary statistics
    overall_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    # Write view-wise statistics
    view_stats = df.groupby('View').agg({
        'Pred MAE': ['mean', 'std'],
        'VQ Pred MAE': ['mean', 'std']
    }).round(6)
    view_stats.to_excel(writer, sheet_name='View Statistics')

print(f"\nResults saved to: {excel_path}")

