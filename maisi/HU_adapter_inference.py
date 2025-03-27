import os
import torch
import numpy as np
import nibabel as nib
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

# List of training and testing cases
train_case_name_list = [
    'E4242', 'E4275', 'E4298', 'E4313',
    'E4245', 'E4276', 'E4299', 'E4317', 'E4246',
    'E4280', 'E4300', 'E4318', 'E4247', 'E4282',
    'E4301', 'E4324', 'E4248', 'E4283', 'E4302',
    'E4325', 'E4250', 'E4284', 'E4306', 'E4328',
    'E4252', 'E4288', 'E4307', 'E4332', 'E4259',
    'E4308', 'E4335', 'E4260', 'E4290', 'E4309',
    'E4336', 'E4261', 'E4292', 'E4310', 'E4337',
    'E4273', 'E4297', 'E4312', 'E4338',
]

test_case_name_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139',
]

# Path helper functions
def get_ct_path(case_name):
    """Get the path to a CT image."""
    return f"NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"

def get_sct_path(case_name):
    """Get the path to a synthetic CT image."""
    return f"NAC_CTAC_Spacing15/CTAC_{case_name}_TS_MAISI.nii.gz"

# Create output root directory
root_dir = "HU_adapter_UNet"
os.makedirs(root_dir, exist_ok=True)

# Set up logging
log_dir = os.path.join(root_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

print(f"Starting inference at {datetime.now()}")
print(f"Log file: {log_file}")

# Set deterministic inference for reproducibility
set_determinism(seed=0)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for inference
inference_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Spacing(
        pixdim=(1.5, 1.5, 1.5),
        mode="bilinear",
    ),
    ScaleIntensity(
        minv=-1024.0,
        maxv=1976.0,
        a_min=0.0,
        a_max=1.0,
        b_min=0.0,
        b_max=1.0,
    ),
    ToTensor(),
])

# Define HU scaler - to convert normalized values back to HU
def normalize_to_hu(normalized_values):
    return normalized_values * (1976.0 - (-1024.0)) + (-1024.0)

# Create model with same architecture as training
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Load the best model from cross-validation
# First find best fold
base_dir = root_dir
folds = [f"fold_{i}" for i in range(1, 5)]
available_folds = [fold for fold in folds if os.path.exists(os.path.join(base_dir, fold, "best_model.pth"))]
best_mae = float('inf')
best_fold = None

for fold in available_folds:
    checkpoint_path = os.path.join(base_dir, fold, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mae_hu = checkpoint.get("mae_hu", float('inf'))
    if mae_hu < best_mae:
        best_mae = mae_hu
        best_fold = fold

if best_fold is None:
    print("No trained models found. Please run training first.")
    log_f.close()
    print = original_print
    exit(1)

print(f"Loading best model from {best_fold} with MAE: {best_mae:.4f} HU")
checkpoint_path = os.path.join(base_dir, best_fold, "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Create output directory
output_dir = os.path.join(root_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

# Run inference on test cases
valid_test_cases = 0
with torch.no_grad():
    for case_name in test_case_name_list:
        print(f"Processing case: {case_name}")
        
        # Load the CT image
        ct_path = get_ct_path(case_name)
        if not os.path.exists(ct_path):
            print(f"  Skipping {case_name}: file not found at {ct_path}")
            continue
            
        # Get original CT image for reference (to preserve metadata)
        original_ct = nib.load(ct_path)
        
        # Transform the image
        try:
            ct_img = inference_transforms(ct_path)
            ct_img = ct_img.unsqueeze(0).to(device)  # Add batch dimension
            
            # Run inference with sliding window
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            predicted_output = sliding_window_inference(
                ct_img, roi_size, sw_batch_size, model, overlap=0.5
            )
            
            # Convert normalized output back to HU range
            predicted_numpy = predicted_output[0, 0].cpu().numpy()
            predicted_hu = normalize_to_hu(predicted_numpy)
            
            # Save the prediction using original affine matrix from CT
            output_nifti = nib.Nifti1Image(predicted_hu, original_ct.affine, original_ct.header)
            output_path = os.path.join(output_dir, f"CTAC_{case_name}_predicted.nii.gz")
            nib.save(output_nifti, output_path)
            
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
    
    for case_name in test_case_name_list:
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
                pred_norm = (pred_img - (-1024.0)) / (1976.0 - (-1024.0))
                gt_norm = (gt_img - (-1024.0)) / (1976.0 - (-1024.0))
                
                pred_tensor = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0).float()
                gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0).float()
                
                # Calculate normalized MAE
                mae_metric(y_pred=pred_tensor, y=gt_tensor)
                mae_norm = mae_metric.aggregate().item()
                
                # Convert MAE back to HU units
                mae_hu = mae_norm * (1976.0 - (-1024.0))
                
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
        
        print(f"\nOverall results:")
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