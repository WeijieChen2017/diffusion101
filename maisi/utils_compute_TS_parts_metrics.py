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

# Mapping dictionary (unchanged)
T2M_mapping = {
    1: 3,
    2: 5,
    3: 14,
    4: 10,
    5: 1,
    6: 12,
    7: 4,
    8: 8,
    9: 9,
    10: 28,
    11: 29,
    12: 30,
    13: 31,
    14: 32,
    15: 11,
    16: 57,
    17: 126,
    18: 19,
    19: 13,
    20: 62,
    21: 15,
    22: 118,
    23: 116,
    24: 117,
    25: 97,
    26: 127,
    27: 33,
    28: 34,
    29: 35,
    30: 36,
    31: 37,
    32: 38,
    33: 39,
    34: 40,
    35: 41,
    36: 42,
    37: 43,
    38: 44,
    39: 45,
    40: 46,
    41: 47,
    42: 48,
    43: 49,
    44: 50,
    45: 51,
    46: 52,
    47: 53,
    48: 54,
    49: 55,
    50: 56,
    51: 115,
    52: 6,
    53: 119,
    54: 109,
    55: 123,
    56: 124,
    57: 112,
    58: 113,
    59: 110,
    60: 111,
    61: 108,
    62: 125,
    63: 7,
    64: 17,
    65: 58,
    66: 59,
    67: 60,
    68: 61,
    69: 87,
    70: 88,
    71: 89,
    72: 90,
    73: 91,
    74: 92,
    75: 93,
    76: 94,
    77: 95,
    78: 96,
    79: 121,
    80: 98,
    81: 99,
    82: 100,
    83: 101,
    84: 102,
    85: 103,
    86: 104,
    87: 105,
    88: 106,
    89: 107,
    90: 22,
    91: 120,
    92: 63,
    93: 64,
    94: 65,
    95: 66,
    96: 67,
    97: 68,
    98: 69,
    99: 70,
    100: 71,
    101: 72,
    102: 73,
    103: 74,
    104: 75,
    105: 76,
    106: 77,
    107: 78,
    108: 79,
    109: 80,
    110: 81,
    111: 82,
    112: 83,
    113: 84,
    114: 85,
    115: 86,
    116: 122,
    117: 114
}

def convert_class_maps_using_T2M():
    """Convert the class maps from map_to_binary using T2M_mapping and print the results"""
    # from .map_to_binary import class_map_5_parts
    
    converted_maps = {}
    
    # Convert and print each part map
    for part_name, part_map in class_map_5_parts.items():
        converted_part = {}
        
        print(f"\n{part_name}:")
        print("-" * 50)
        
        # Convert each class ID using T2M_mapping
        for orig_id, class_name in part_map.items():
            if orig_id in T2M_mapping:
                new_id = T2M_mapping[orig_id]
                converted_part[new_id] = class_name
                print(f"Original ID: {orig_id:3d} -> New ID: {new_id:3d} | {class_name}")
            else:
                print(f"Warning: ID {orig_id} not found in T2M_mapping | {class_name}")
                converted_part[orig_id] = class_name
                
        converted_maps[part_name] = converted_part
        
    return converted_maps

def compute_parts_metrics():
    """Compute MAE metrics (mean and std) for both original and adjusted synthetic CTs"""
    
    # Initialize metrics dictionary for both predictions
    metrics_dict = {
        "original": {
            "mae_by_part": {},
            "mae_std_by_part": {},  # Added std metrics
            "mae_class_200": 0,
            "mae_std_class_200": 0  # Added std for class 200
        },
        "adjusted": {
            "mae_by_part": {},
            "mae_std_by_part": {},  # Added std metrics
            "mae_class_200": 0,
            "mae_std_class_200": 0  # Added std for class 200
        }
    }
    
    root_dir = "NAC_CTAC_Spacing15"
    synCT_dir = f"{root_dir}/inference_20250128_noon"
    
    # Convert class maps using T2M mapping
    converted_maps = convert_class_maps_using_T2M()
    
    # Initialize lists to store MAE values for each prediction
    original_mae_values = {part_name: [] for part_name in converted_maps.keys()}
    adjusted_mae_values = {part_name: [] for part_name in converted_maps.keys()}
    original_class_200_values = []
    adjusted_class_200_values = []
    
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
        for part_name, part_map in converted_maps.items():
            part_mask = np.zeros_like(seg_data, dtype=bool)
            
            # Create mask for current part
            for class_id in part_map.keys():
                part_mask |= (seg_data == class_id)
                
            # Compute MAE for both predictions
            if np.any(part_mask):
                # Original prediction
                mae_orig = np.mean(np.abs(ct_data[part_mask] - synCT_data[part_mask]))
                original_mae_values[part_name].append(mae_orig)
                
                # Adjusted prediction
                mae_adj = np.mean(np.abs(ct_data[part_mask] - synCT_adjusted_data[part_mask]))
                adjusted_mae_values[part_name].append(mae_adj)
                
                print(f"{part_name}:")
                print(f"  Original MAE: {mae_orig:.4f}")
                print(f"  Adjusted MAE: {mae_adj:.4f}")
            else:
                print(f"Warning: No voxels found for {part_name} in {case_name}")
        
        # Compute MAE for class 200 for both predictions
        class_200_mask = seg_data == 200
        if np.any(class_200_mask):
            # Original prediction
            mae_200_orig = np.mean(np.abs(ct_data[class_200_mask] - synCT_data[class_200_mask]))
            original_class_200_values.append(mae_200_orig)
            
            # Adjusted prediction
            mae_200_adj = np.mean(np.abs(ct_data[class_200_mask] - synCT_adjusted_data[class_200_mask]))
            adjusted_class_200_values.append(mae_200_adj)
            
            print(f"Class 200:")
            print(f"  Original MAE: {mae_200_orig:.4f}")
            print(f"  Adjusted MAE: {mae_200_adj:.4f}")
        else:
            print(f"Warning: No voxels found for class 200 in {case_name}")
    
    # After collecting all values, compute mean and std
    for part_name in converted_maps.keys():
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
    
    # Save metrics to json
    metrics_json_path = f"{root_dir}/parts_mae_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f)
    
    # Print results
    for pred_type in ["original", "adjusted"]:
        print(f"\nMetrics for {pred_type} prediction:")
        for part_name in converted_maps.keys():
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