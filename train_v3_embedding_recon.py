import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
                padded_shape[2] // 4  # z
            ),
            "sagittal": (
                padded_shape[0],  # x (unchanged)
                padded_shape[1] // 4,  # y (zoomed)
                padded_shape[2] // 4  # z
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

def get_embedding_dimensions(orientation, flattened, expected_shapes):
    """
    Get correct embedding dimensions based on the pre-computed expected shapes.
    """
    width = expected_shapes[orientation][1]   # Second dimension from expected shape
    height = expected_shapes[orientation][2]  # Third dimension from expected shape
    
    # Verify the dimensions match the flattened size
    if width * height != flattened:
        raise ValueError(f"Computed dimensions {width}x{height} != flattened size {flattened}")
    
    return width, height

def visualize_and_save_embeddings(data_div, vq_weights_path="James_data_v3/vq_f4_weights_attn.npy"):
    """
    Convert indices to embeddings, normalize them, and create visualizations with PET/CT.
    """
    # Load VQ weights dictionary
    print(f"Loading VQ weights from {vq_weights_path}")
    vq_weights = np.load(vq_weights_path)  # Shape: (8192, 3)
    
    # First get shape information
    shape_info = collect_and_save_all_shapes(data_div)
    
    # Create output directories
    output_dir = "data_inspection/embeddings"
    vis_dir = "data_inspection/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each case
    for cv_idx in range(5):
        for hashname in tqdm(data_div[f"cv{cv_idx}"], desc=f"Processing CV{cv_idx}"):
            try:
                expected_shapes = shape_info[hashname]["volume"]["expected_index_shape"]
                
                # Process each orientation
                for orientation in ["axial", "sagittal", "coronal"]:
                    # Load PET and CT indices
                    pet_index_path = f"James_data_v3/index/{hashname}_x_{orientation}_ind.npy"
                    ct_index_path = f"James_data_v3/index/{hashname}_y_{orientation}_ind.npy"
                    
                    try:
                        # Process PET indices
                        pet_indices = np.load(pet_index_path)
                        n_slices, flattened = pet_indices.shape
                        width, height = get_embedding_dimensions(orientation, flattened, expected_shapes)
                        
                        # Process CT indices
                        ct_indices = np.load(ct_index_path)
                        
                        # Get embeddings for both PET and CT
                        # PET embeddings
                        pet_indices_reshaped = pet_indices.reshape(n_slices, width, height)
                        pet_embeddings = vq_weights[pet_indices_reshaped.flatten()].reshape(n_slices, width, height, 3)
                        pet_embeddings = pet_embeddings.transpose(0, 3, 1, 2)
                        pet_embeddings_norm = pet_embeddings / 8.0 + 0.5
                        
                        # CT embeddings
                        ct_indices_reshaped = ct_indices.reshape(n_slices, width, height)
                        ct_embeddings = vq_weights[ct_indices_reshaped.flatten()].reshape(n_slices, width, height, 3)
                        ct_embeddings = ct_embeddings.transpose(0, 3, 1, 2)
                        ct_embeddings_norm = ct_embeddings / 8.0 + 0.5
                        
                        # Save normalized embeddings
                        output_path = os.path.join(output_dir, f"{hashname}_{orientation}_embedding_norm.npz")
                        np.savez_compressed(
                            output_path,
                            pet_embedding=pet_embeddings_norm,
                            ct_embedding=ct_embeddings_norm,
                            shape=pet_embeddings_norm.shape
                        )
                        
                        # Create visualizations for every 10th slice
                        for slice_idx in range(0, n_slices, 10):
                            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                            fig.suptitle(f"{hashname} - {orientation} - Slice {slice_idx}/{n_slices}")
                            
                            # Get embedding slices (both original and normalized)
                            pet_emb_slice_orig = pet_embeddings[slice_idx].transpose(1, 2, 0)  # (H, W, 3)
                            pet_emb_slice_norm = pet_embeddings_norm[slice_idx].transpose(1, 2, 0)
                            ct_emb_slice_orig = ct_embeddings[slice_idx].transpose(1, 2, 0)
                            ct_emb_slice_norm = ct_embeddings_norm[slice_idx].transpose(1, 2, 0)
                            
                            # Plot PET row
                            # Original indices
                            axes[0, 0].imshow(pet_indices_reshaped[slice_idx], cmap='gray')
                            axes[0, 0].set_title('PET Indices')
                            axes[0, 0].axis('off')
                            
                            # PET embedding (RGB)
                            axes[0, 1].imshow(pet_emb_slice_norm)
                            axes[0, 1].set_title('PET Embedding')
                            axes[0, 1].axis('off')
                            
                            # PET embedding histogram (original values)
                            axes[0, 2].hist(pet_emb_slice_orig.ravel(), bins=50, color='blue', alpha=0.7)
                            axes[0, 2].set_yscale('log')
                            axes[0, 2].set_title('PET Embedding Distribution\n(Original [-4, 4])')
                            axes[0, 2].set_xlabel('Value')
                            axes[0, 2].set_ylabel('Count (log)')
                            axes[0, 2].set_xlim(-4, 4)
                            
                            # Plot CT row
                            # Original indices
                            axes[1, 0].imshow(ct_indices_reshaped[slice_idx], cmap='gray')
                            axes[1, 0].set_title('CT Indices')
                            axes[1, 0].axis('off')
                            
                            # CT embedding (RGB)
                            axes[1, 1].imshow(ct_emb_slice_norm)
                            axes[1, 1].set_title('CT Embedding')
                            axes[1, 1].axis('off')
                            
                            # CT embedding histogram (original values)
                            axes[1, 2].hist(ct_emb_slice_orig.ravel(), bins=50, color='red', alpha=0.7)
                            axes[1, 2].set_yscale('log')
                            axes[1, 2].set_title('CT Embedding Distribution\n(Original [-4, 4])')
                            axes[1, 2].set_xlabel('Value')
                            axes[1, 2].set_ylabel('Count (log)')
                            axes[1, 2].set_xlim(-4, 4)
                            
                            # Save figure
                            plt.tight_layout()
                            vis_path = os.path.join(vis_dir, f"{hashname}_{orientation}_slice_{slice_idx:03d}.png")
                            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                        print(f"Processed {hashname} {orientation}:")
                        print(f"  Saved embeddings to: {output_path}")
                        print(f"  Saved visualizations to: {vis_dir}")
                        
                    except Exception as e:
                        print(f"Error processing {hashname} {orientation}: {str(e)}")
                        
            except Exception as e:
                print(f"Error loading for {hashname}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    # Load data_div from JSON file
    with open("James_data_v3/cv_list.json", "r") as f:
        data_div = json.load(f)
    
    # First collect and analyze shapes
    shape_info = collect_and_save_all_shapes(data_div)
    mismatches = analyze_shape_predictions(shape_info)
    
    # Then convert indices to embeddings and create visualizations
    visualize_and_save_embeddings(data_div)