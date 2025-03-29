import os
import torch
import numpy as np
import nibabel as nib
import argparse
import json
from datetime import datetime
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor,
    Spacing,
)
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# Import common utilities
from maisi.HU_adapter_common import (
    ROOT_DIR, TRAIN_CASES, TEST_CASES, HU_MIN, HU_MAX,
    get_ct_path, get_sct_path, get_folds_path, get_fold_dir
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run inference for a specific fold')
parser.add_argument('--fold', type=int, required=True, help='Fold number (1-4)')
parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
args = parser.parse_args()

# Set up logging
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_fold{args.fold}_gpu{args.gpu}_detailed.log")
log_f = open(log_file, 'w')

# Define a custom print function to log all output
original_print = print
def custom_print(*args, **kwargs):
    output = " ".join(map(str, args))
    original_print(output, **kwargs)
    log_f.write(output + "\n")
    log_f.flush()

# Replace the print function with our custom one
print = custom_print

print(f"Starting inference for fold {args.fold} on GPU {args.gpu} at {datetime.now()}")
print(f"Log file: {log_file}")

# Set deterministic inference for reproducibility
set_determinism(seed=args.fold)  # Use different seed for each fold

# Set device based on argument
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} (GPU {args.gpu})")

# Load fold data
folds_path = get_folds_path()
print(f"Looking for folds data at: {folds_path}")
if not os.path.exists(folds_path):
    raise FileNotFoundError(f"Folds file not found at {folds_path}. Run HU_adapter_create_folds.py first.")

with open(folds_path, 'r') as f:
    folds = json.load(f)

fold_key = f"fold_{args.fold}"
if fold_key not in folds:
    raise ValueError(f"Invalid fold number: {args.fold}. Available folds: {list(folds.keys())}")

test_cases = folds[fold_key]["test"]
print(f"Running inference for {fold_key}")
print(f"Test cases: {len(test_cases)}")

# Define transforms for inference
inference_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Spacing(
        pixdim=(1.5, 1.5, 1.5),
        mode="bilinear",
    ),
    ScaleIntensity(
        minv=HU_MIN,
        maxv=HU_MAX,
        a_min=0.0,
        a_max=1.0,
        b_min=0.0,
        b_max=1.0,
    ),
    ToTensor(),
])

# Define HU scaler - to convert normalized values back to HU
def normalize_to_hu(normalized_values):
    return normalized_values * (HU_MAX - HU_MIN) + HU_MIN

# Create model with same architecture as training
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=6,
)

# Load the model for this fold
fold_dir = get_fold_dir(args.fold)
checkpoint_path = os.path.join(fold_dir, "best_model.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

print(f"Loading model from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Create output directory for this fold
output_dir = os.path.join(fold_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

# Run inference on test cases
valid_test_cases = 0
with torch.no_grad():
    for case_name in test_cases:
        print(f"Processing case: {case_name}")
        
        # Load the CT image
        ct_path = get_ct_path(case_name)
        if not os.path.exists(ct_path):
            print(f"  Skipping {case_name}: file not found at {ct_path}")
            continue
            
        # Get original CT image for reference (to preserve metadata)
        original_ct = nib.load(ct_path)
        original_data = original_ct.get_fdata()
        print(f"  Original CT range: [{original_data.min():.2f}, {original_data.max():.2f}] HU")
        
        # Transform the image
        try:
            ct_img = inference_transforms(ct_path)
            ct_img = ct_img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Verify normalization
            normalized_data = ct_img[0, 0].cpu().numpy()
            print(f"  Normalized range: [{normalized_data.min():.2f}, {normalized_data.max():.2f}]")
            
            # Run inference with sliding window
            roi_size = (128, 128, 128)
            sw_batch_size = 16
            predicted_output = sliding_window_inference(
                ct_img, roi_size, sw_batch_size, model, overlap=0.25
            )
            
            # Convert normalized output back to HU range
            predicted_numpy = predicted_output[0, 0].cpu().numpy()
            predicted_hu = normalize_to_hu(predicted_numpy)
            
            # Verify denormalization
            print(f"  Predicted range: [{predicted_hu.min():.2f}, {predicted_hu.max():.2f}] HU")
            
            # Save the prediction using original affine matrix from CT
            output_nifti = nib.Nifti1Image(predicted_hu, original_ct.affine, original_ct.header)
            output_path = os.path.join(output_dir, f"CTAC_{case_name}_predicted.nii.gz")
            nib.save(output_nifti, output_path)
            
            # Verify saved file
            saved_nifti = nib.load(output_path)
            saved_data = saved_nifti.get_fdata()
            print(f"  Saved file range: [{saved_data.min():.2f}, {saved_data.max():.2f}] HU")
            print(f"  Saved prediction to {output_path}")
            valid_test_cases += 1
        except Exception as e:
            print(f"  Error processing case {case_name}: {str(e)}")

print(f"Inference completed for {valid_test_cases} test cases")

# Calculate evaluation metrics if ground truth is available
try:
    from monai.metrics import MAEMetric
    import pandas as pd
    
    mae_metric = MAEMetric()
    results = []
    
    for case_name in test_cases:
        # Load prediction
        pred_path = os.path.join(output_dir, f"CTAC_{case_name}_predicted.nii.gz")
        gt_path = get_sct_path(case_name)
        
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            print(f"Evaluating case: {case_name}")
            
            # Load ground truth and prediction
            try:
                pred_img = nib.load(pred_path).get_fdata()
                gt_img = nib.load(gt_path).get_fdata()
                
                # Ensure same dimensions
                if pred_img.shape != gt_img.shape:
                    print(f"  Shape mismatch: pred {pred_img.shape}, gt {gt_img.shape}")
                    continue
                
                # Convert to tensors and normalize for metric calculation 
                # Note: Images are already in HU range, so we normalize to 0-1 for MAE calculation
                pred_norm = (pred_img - HU_MIN) / (HU_MAX - HU_MIN)
                gt_norm = (gt_img - HU_MIN) / (HU_MAX - HU_MIN)
                
                pred_tensor = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0).float()
                gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0).float()
                
                # Calculate normalized MAE
                mae_metric(y_pred=pred_tensor, y=gt_tensor)
                mae_norm = mae_metric.aggregate().item()
                
                # Convert MAE back to HU units
                mae_hu = mae_norm * (HU_MAX - HU_MIN)
                
                results.append({
                    "case_name": case_name,
                    "mae_hu": mae_hu
                })
                
                print(f"  Case {case_name}: MAE = {mae_hu:.4f} HU")
                mae_metric.reset()
            except Exception as e:
                print(f"  Error evaluating case {case_name}: {str(e)}")
    
    # Create summary dataframe and save
    if results:
        df = pd.DataFrame(results)
        mean_mae = df["mae_hu"].mean()
        std_mae = df["mae_hu"].std()
        
        print(f"\nOverall results for fold {args.fold}:")
        print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f} HU")
        
        # Save results to CSV
        results_path = os.path.join(output_dir, "evaluation_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
except Exception as e:
    print(f"Evaluation skipped: {str(e)}")

# Close log file
log_f.close()
# Restore original print
print = original_print 