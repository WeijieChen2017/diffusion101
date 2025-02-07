from map_to_binary import class_map_5_parts
import os
import nibabel as nib
import numpy as np
import json

case_name_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 
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

class_map_5_parts = {
    'organs': [3, 5, 14, 10, 1, 12, 4, 8, 9, 28, 29, 30, 31, 32, 11, 57, 126, 19, 13, 62, 15, 118, 116, 117],
    'vertebrae': [97, 127, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    'cardiac': [115, 6, 119, 109, 123, 124, 112, 113, 110, 111, 108, 125, 7, 17, 58, 59, 60, 61],
    'muscles': [95, 96, 121, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 22, 120],
    'ribs': [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 122, 114],
    'body_contour': [200],
}

def compute_parts_metrics():
    """Compute MAE metrics (mean and std) for both original and adjusted synthetic CTs"""
    
    # Initialize metrics dictionary for both predictions
    metrics_dict = {
        "original": {
            "mae_by_part": {},
            "mae_std_by_part": {},
            "mae_class_200": 0,
            "mae_std_class_200": 0,
            "raw_data": {
                "mae_by_part": {},
                "mae_class_200": []
            }
        },
        "adjusted": {
            "mae_by_part": {},
            "mae_std_by_part": {},
            "mae_class_200": 0,
            "mae_std_class_200": 0,
            "raw_data": {
                "mae_by_part": {},
                "mae_class_200": []
            }
        }
    }
    
    root_dir = "NAC_CTAC_Spacing15"
    synCT_dir = f"{root_dir}/inference_20250128_noon"
    
    # Initialize lists to store MAE values for each prediction
    original_mae_values = {part_name: [] for part_name in class_map_5_parts.keys()}
    adjusted_mae_values = {part_name: [] for part_name in class_map_5_parts.keys()}
    original_class_200_values = []
    adjusted_class_200_values = []
    
    # Initialize raw data storage
    for part_name in class_map_5_parts.keys():
        metrics_dict["original"]["raw_data"]["mae_by_part"][part_name] = {}
        metrics_dict["adjusted"]["raw_data"]["mae_by_part"][part_name] = {}
    
    for case_name in case_name_list:
        # Load CT and both synCT predictions
        ct_path = f"{root_dir}/CTAC_{case_name}_cropped.nii.gz"
        synCT_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI.nii.gz"
        synCT_adjusted_path = f"{synCT_dir}/CTAC_{case_name}_TS_MAISI_adjusted.nii.gz"
        seg_path = f"{root_dir}/CTAC_{case_name}_TS_label.nii.gz"
        
        ct_data = nib.load(ct_path).get_fdata()
        synCT_data = nib.load(synCT_path).get_fdata()
        synCT_adjusted_data = nib.load(synCT_adjusted_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
        
        # Handle padding/cropping for both predictions
        if ct_data.shape[2] > synCT_data.shape[2]:
            pad_size = ((0, 0), (0, 0), (0, ct_data.shape[2] - synCT_data.shape[2]))
            synCT_data = np.pad(synCT_data, pad_size, mode="constant", constant_values=-1024)
            synCT_adjusted_data = np.pad(synCT_adjusted_data, pad_size, mode="constant", constant_values=-1024)
        else:
            synCT_data = synCT_data[:, :, :ct_data.shape[2]]
            synCT_adjusted_data = synCT_adjusted_data[:, :, :ct_data.shape[2]]
        
        print(f"\nProcessing {case_name}:")
        print("-" * 50)
        
        # Compute MAE for each part for both predictions
        for part_name, class_ids in class_map_5_parts.items():
            part_mask = np.zeros_like(seg_data, dtype=bool)
            
            # Create mask for current part using the class IDs list
            for class_id in class_ids:
                part_mask |= (seg_data == class_id)
                
            # Compute MAE for both predictions
            if np.any(part_mask):
                # Original prediction
                mae_orig = np.mean(np.abs(ct_data[part_mask] - synCT_data[part_mask]))
                original_mae_values[part_name].append(mae_orig)
                metrics_dict["original"]["raw_data"]["mae_by_part"][part_name][case_name] = mae_orig
                
                # Adjusted prediction
                mae_adj = np.mean(np.abs(ct_data[part_mask] - synCT_adjusted_data[part_mask]))
                adjusted_mae_values[part_name].append(mae_adj)
                metrics_dict["adjusted"]["raw_data"]["mae_by_part"][part_name][case_name] = mae_adj
                
                print(f"{part_name}:")
                print(f"  Original MAE: {mae_orig:.4f}")
                print(f"  Adjusted MAE: {mae_adj:.4f}")
            else:
                print(f"Warning: No voxels found for {part_name} in {case_name}")
        
        # Compute MAE for class 200 (body_contour)
        class_200_mask = seg_data == 200
        if np.any(class_200_mask):
            mae_200_orig = np.mean(np.abs(ct_data[class_200_mask] - synCT_data[class_200_mask]))
            mae_200_adj = np.mean(np.abs(ct_data[class_200_mask] - synCT_adjusted_data[class_200_mask]))
            
            metrics_dict["original"]["raw_data"]["mae_class_200"].append({
                "case": case_name,
                "mae": mae_200_orig
            })
            metrics_dict["adjusted"]["raw_data"]["mae_class_200"].append({
                "case": case_name,
                "mae": mae_200_adj
            })
            
            original_class_200_values.append(mae_200_orig)
            adjusted_class_200_values.append(mae_200_adj)
            
            print(f"Body contour (class 200):")
            print(f"  Original MAE: {mae_200_orig:.4f}")
            print(f"  Adjusted MAE: {mae_200_adj:.4f}")
        else:
            print(f"Warning: No voxels found for body contour in {case_name}")
    
    # After collecting all values, compute mean and std
    for part_name in class_map_5_parts.keys():
        # Original prediction
        if original_mae_values[part_name]:
            metrics_dict["original"]["mae_by_part"][part_name] = np.mean(original_mae_values[part_name])
            metrics_dict["original"]["mae_std_by_part"][part_name] = np.std(original_mae_values[part_name])
        else:
            metrics_dict["original"]["mae_by_part"][part_name] = 0
            metrics_dict["original"]["mae_std_by_part"][part_name] = 0
            print(f"Warning: No valid MAE values for {part_name} in original prediction")
        
        # Adjusted prediction
        if adjusted_mae_values[part_name]:
            metrics_dict["adjusted"]["mae_by_part"][part_name] = np.mean(adjusted_mae_values[part_name])
            metrics_dict["adjusted"]["mae_std_by_part"][part_name] = np.std(adjusted_mae_values[part_name])
        else:
            metrics_dict["adjusted"]["mae_by_part"][part_name] = 0
            metrics_dict["adjusted"]["mae_std_by_part"][part_name] = 0
            print(f"Warning: No valid MAE values for {part_name} in adjusted prediction")
    
    # Compute mean and std for class 200
    if original_class_200_values:
        metrics_dict["original"]["mae_class_200"] = np.mean(original_class_200_values)
        metrics_dict["original"]["mae_std_class_200"] = np.std(original_class_200_values)
    if adjusted_class_200_values:
        metrics_dict["adjusted"]["mae_class_200"] = np.mean(adjusted_class_200_values)
        metrics_dict["adjusted"]["mae_std_class_200"] = np.std(adjusted_class_200_values)
    
    # Save both summary metrics and raw data to json
    metrics_json_path = f"{root_dir}/parts_mae_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)  # Added indent for better readability
    
    # Print results
    for pred_type in ["original", "adjusted"]:
        print(f"\nMetrics for {pred_type} prediction:")
        for part_name in class_map_5_parts.keys():
            mean = metrics_dict[pred_type]["mae_by_part"][part_name]
            std = metrics_dict[pred_type]["mae_std_by_part"][part_name]
            print(f"{part_name}:")
            print(f"  Mean MAE: {mean:.4f} ± {std:.4f}")
        
        mean_200 = metrics_dict[pred_type]["mae_class_200"]
        std_200 = metrics_dict[pred_type]["mae_std_class_200"]
        print(f"Class 200:")
        print(f"  Mean MAE: {mean_200:.4f} ± {std_200:.4f}")
    
    print(f"\nSaved metrics to {metrics_json_path}")
    return metrics_dict

# Run the computation
metrics = compute_parts_metrics()