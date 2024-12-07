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
        # Format: (unchanged_dim, width, height)
        expected_shapes = {
            "axial": (
                padded_shape[2],  # z (unchanged)
                padded_shape[0] // 4,  # x (zoomed)
                padded_shape[1] // 4   # y (zoomed)
            ),
            "coronal": (
                padded_shape[1],  # y (unchanged)
                padded_shape[0] // 4,  # x (zoomed)
                padded_shape[2]   # z
            ),
            "sagittal": (
                padded_shape[0],  # x (unchanged)
                padded_shape[1] // 4,  # y (zoomed)
                padded_shape[2]   # z
            )
        }
        
        # Initialize shape info for this case
        shape_info[hashname] = {
            "volume": {
                "original_shape": orig_shape,
                "padded_shape": padded_shape,
                "expected_index_shape": expected_shapes
            },
            "index": {}
        }
        
        # Load and store index shapes
        for orientation, path in index_paths.items():
            try:
                index_data = np.load(path)
                original_shape = index_data.shape
                
                # Expected flattened shape (c, w*h)
                expected_c = expected_shapes[orientation][0]
                expected_w = expected_shapes[orientation][1]
                expected_h = expected_shapes[orientation][2]
                expected_flat_shape = (expected_c, expected_w * expected_h)
                
                # Check if shape matches expected and can be reshaped
                shape_matches = (
                    len(index_data.shape) == 2 and
                    index_data.shape[0] == expected_c and
                    index_data.shape[1] == (expected_w * expected_h)
                )
                
                shape_info[hashname]["index"][orientation] = {
                    "original_shape": original_shape,
                    "expected_flat_shape": expected_flat_shape,
                    "expected_reshaped": (expected_c, expected_w, expected_h),
                    "shape_matches": shape_matches,
                    "dtype": str(index_data.dtype),
                    "min": int(index_data.min()),
                    "max": int(index_data.max())
                }
                
                if not shape_matches:
                    shape_info[hashname]["index"][orientation]["error"] = (
                        f"Shape mismatch: got {original_shape}, "
                        f"expected {expected_flat_shape} (to be reshaped to {(expected_c, expected_w, expected_h)})"
                    )
                
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
            if "error" in info["index"][orientation]:
                case_mismatches[orientation] = {
                    "expected_flat": info["volume"]["expected_index_shape"][orientation],
                    "error": info["index"][orientation]["error"]
                }
                case_has_mismatch = True
                continue
            
            if not info["index"][orientation]["shape_matches"]:
                case_mismatches[orientation] = {
                    "original_shape": info["index"][orientation]["original_shape"],
                    "expected_flat": info["index"][orientation]["expected_flat_shape"],
                    "expected_reshaped": info["index"][orientation]["expected_reshaped"]
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
                print(f"  {orientation}:")
                if "error" in mismatch_info:
                    print(f"    Error: {mismatch_info['error']}")
                else:
                    print(f"    Got shape: {mismatch_info['original_shape']}")
                    print(f"    Expected flat: {mismatch_info['expected_flat']}")
                    print(f"    Should reshape to: {mismatch_info['expected_reshaped']}")
    
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