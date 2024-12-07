import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import json

def collect_and_save_all_shapes(data_div):
    """
    Collect shape information for all volumes and their corresponding index files.
    """
    # Create data inspection directory
    root = "data_inspection"
    os.makedirs(root, exist_ok=True)
    
    cv_all = range(5)
    all_cases = []
    shape_info = {}
    
    # Collect all cases from all CVs
    for cv_idx in cv_all:
        all_cases.extend(data_div[f"cv{cv_idx}"])
    
    print(f"Processing {len(all_cases)} cases...")
    
    for hashname in tqdm(all_cases):
        # Define volume paths
        pet_path = f"James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz"
        ct_path = f"James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz"
        
        # Define index paths
        index_paths = {
            "axial": f"James_data_v3/index/{hashname}_x_axial_ind.npy",
            "sagittal": f"James_data_v3/index/{hashname}_x_sagittal_ind.npy",
            "coronal": f"James_data_v3/index/{hashname}_x_coronal_ind.npy"
        }
        
        # Load volumes and get shapes
        pet_vol = nib.load(pet_path)
        ct_vol = nib.load(ct_path)
        
        orig_shape = pet_vol.shape
        padded_shape = (
            orig_shape[0],
            orig_shape[1],
            orig_shape[2] + (4 - orig_shape[2] % 4) % 4
        )
        
        # Calculate expected shapes after zoom and transpose
        expected_shapes = {
            "axial": (
                padded_shape[2],  # z (unchanged)
                padded_shape[0] // 4,  # x (zoomed)
                padded_shape[1] // 4   # y (zoomed)
            ),
            "coronal": (
                padded_shape[1],  # y (unchanged)
                padded_shape[2],  # z
                padded_shape[0] // 4   # x (zoomed)
            ),
            "sagittal": (
                padded_shape[0],  # x (unchanged)
                padded_shape[2],  # z
                padded_shape[1] // 4   # y (zoomed)
            )
        }
        
        # Initialize shape info for this case
        shape_info[hashname] = {
            "volume": {
                "original_shape": {
                    "axial": orig_shape,
                    "sagittal": (orig_shape[2], orig_shape[0], orig_shape[1]),
                    "coronal": (orig_shape[2], orig_shape[1], orig_shape[0])
                },
                "padded_shape": {
                    "axial": padded_shape,
                    "sagittal": (padded_shape[2], padded_shape[0], padded_shape[1]),
                    "coronal": (padded_shape[2], padded_shape[1], padded_shape[0])
                },
                "expected_index_shape": expected_shapes
            },
            "index": {}
        }
        
        # Load and store index shapes
        for orientation, path in index_paths.items():
            try:
                index_data = np.load(path)
                shape_info[hashname]["index"][orientation] = {
                    "shape": index_data.shape,
                    "dtype": str(index_data.dtype),
                    "min": int(index_data.min()),
                    "max": int(index_data.max()),
                    "matches_expected": (index_data.shape == expected_shapes[orientation])
                }
            except Exception as e:
                print(f"Error loading index file for {hashname} {orientation}: {str(e)}")
                shape_info[hashname]["index"][orientation] = {
                    "error": str(e)
                }
    
    # Save to JSON file
    output_path = os.path.join(root, "all_shapes.json")
    with open(output_path, "w") as f:
        json.dump(shape_info, f, indent=4)
    
    print(f"Shape information saved to {output_path}")
    return shape_info

def analyze_shape_predictions(shape_info):
    """
    Analyze and print discrepancies between predicted and actual shapes.
    """
    print("\nShape Analysis Report:")
    print("=" * 80)
    
    mismatches = {}
    total_cases = len(shape_info)
    mismatch_count = 0
    
    for hashname, info in shape_info.items():
        case_has_mismatch = False
        case_mismatches = {}
        
        for orientation in ["axial", "sagittal", "coronal"]:
            expected_shape = info["volume"]["expected_index_shape"][orientation]
            
            if "error" in info["index"][orientation]:
                case_mismatches[orientation] = {
                    "expected": expected_shape,
                    "actual": "File Error",
                    "error": info["index"][orientation]["error"]
                }
                case_has_mismatch = True
                continue
                
            actual_shape = info["index"][orientation]["shape"]
            
            if actual_shape != expected_shape:
                case_mismatches[orientation] = {
                    "expected": expected_shape,
                    "actual": actual_shape
                }
                case_has_mismatch = True
        
        if case_has_mismatch:
            mismatches[hashname] = case_mismatches
            mismatch_count += 1
    
    # Print summary
    print(f"Total cases analyzed: {total_cases}")
    print(f"Cases with mismatches: {mismatch_count}")
    print(f"Match rate: {((total_cases - mismatch_count) / total_cases) * 100:.2f}%")
    
    # Print detailed mismatches
    if mismatches:
        print("\nDetailed Mismatches:")
        print("-" * 80)
        for hashname, case_mismatches in mismatches.items():
            print(f"\nCase: {hashname}")
            for orientation, mismatch_info in case_mismatches.items():
                if "error" in mismatch_info:
                    print(f"  {orientation}: File Error - {mismatch_info['error']}")
                else:
                    print(f"  {orientation}:")
                    print(f"    Expected: {mismatch_info['expected']}")
                    print(f"    Actual:   {mismatch_info['actual']}")
    
    return mismatches

# Example usage:
if __name__ == "__main__":
    # Load data_div from JSON file
    with open("James_data_v3/cv_list.json", "r") as f:
        data_div = json.load(f)
    
    # Collect shape information
    shape_info = collect_and_save_all_shapes(data_div)
    
    # Analyze shapes
    mismatches = analyze_shape_predictions(shape_info)