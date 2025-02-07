import numpy as np
import json
import pandas as pd
import os
from glob import glob
from .map_to_binary import class_map_5_parts
# from .utils_compute_TS_parts_metrics import T2M_mapping, convert_class_maps_using_T2M

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


def compute_HU_difference_by_part():
    """
    Compute HU value distributions for each part using the class_map_5_parts mapping, 
    saving statistics to separate CSV files
    """
    # Initialize dictionaries to store HU values for each part
    part_hu_values = {
        part_name: {
            'CT_values': [],
            'sCT_values': []
        } for part_name in class_map_5_parts.keys()
    }

    # Directory path
    data_dir = "playground/HU_dist"
    
    # Get list of all HU stats files
    stats_pattern = f"{data_dir}/SynthRad_*_HU_stats.npy"
    stats_files = glob(stats_pattern)

    print("Processing HU distribution files...")
    for stats_file in stats_files:
        print(f"Processing {os.path.basename(stats_file)}...")
        
        # Load the HU statistics data
        hu_stats = np.load(stats_file, allow_pickle=True).item()
        
        # Process each class ID in the HU stats
        for class_id_str in hu_stats.keys():
            class_id = int(float(class_id_str))  # Convert string key to int
            
            # Find which part this class belongs to
            part_name = None
            for name, class_ids in class_map_5_parts.items():
                if class_id in class_ids:
                    part_name = name
                    break
            
            if part_name is not None:
                # Extract CT and sCT values
                ct_values = hu_stats[class_id_str]['CT_values']
                sct_values = hu_stats[class_id_str]['sCT_values']
                
                # Add to the corresponding part's collection
                part_hu_values[part_name]['CT_values'].extend(ct_values)
                part_hu_values[part_name]['sCT_values'].extend(sct_values)

    # Calculate statistics and save to CSV files
    output_dir = f"{data_dir}/part_statistics"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving statistics to CSV files...")
    for part_name, values in part_hu_values.items():
        ct_values = np.array(values['CT_values'])
        sct_values = np.array(values['sCT_values'])
        
        if len(ct_values) == 0:
            print(f"Warning: No values found for {part_name}")
            continue
        
        # Create statistics dictionary
        stats = {
            'Metric': ['Mean', 'Std', 'Min', 'Max', 'P25', 'P50', 'P75'],
            'CT': [
                np.mean(ct_values),
                np.std(ct_values),
                np.min(ct_values),
                np.max(ct_values),
                np.percentile(ct_values, 25),
                np.percentile(ct_values, 50),
                np.percentile(ct_values, 75)
            ],
            'sCT': [
                np.mean(sct_values),
                np.std(sct_values),
                np.min(sct_values),
                np.max(sct_values),
                np.percentile(sct_values, 25),
                np.percentile(sct_values, 50),
                np.percentile(sct_values, 75)
            ],
            'Difference': [
                np.mean(sct_values - ct_values),
                np.std(sct_values - ct_values),
                np.min(sct_values - ct_values),
                np.max(sct_values - ct_values),
                np.percentile(sct_values - ct_values, 25),
                np.percentile(sct_values - ct_values, 50),
                np.percentile(sct_values - ct_values, 75)
            ]
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(stats)
        csv_path = f"{output_dir}/{part_name}_HU_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved statistics for {part_name} to {csv_path}")

        # Also save raw HU values for potential further analysis
        raw_values = {
            'CT': ct_values,
            'sCT': sct_values,
            'Difference': sct_values - ct_values
        }
        np_path = f"{output_dir}/{part_name}_raw_values.npy"
        np.save(np_path, raw_values)
        print(f"Saved raw values for {part_name} to {np_path}")

if __name__ == "__main__":
    compute_HU_difference_by_part()
