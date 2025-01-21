# seg_map_folder = "CTACIVV_MAISeg" # 512, 512, 335 1.367, 1.367, 3.27
# body_contour_folder = "../James_data_v3/mask"
# find every .nii.gz file in this folder

# load all files in the target folder
import os
import glob
import nibabel as nib
import numpy as np

# file_list = sorted(glob.glob(body_contour_folder+"/mask_body_contour_*.nii.gz"))
# print(f"We find {len(file_list)} files in the target folder")
# for file in file_list:
#     new_name = file.replace(".nii.gz", "_Spacing15.nii.gz")
#     command = f"3dresample -dxyz 1.5 1.5 1.5 -rmode NN -prefix {new_name} -input {file}"
#     print(command)

# folder = "Spacing15"
# seg_file_list = sorted(glob.glob(folder+"/CTACIVV_*_MAISI_Spacing15.nii.gz"))
# for seg_path in seg_file_list:
#     case_name = os.path.basename(seg_path)[8:13]
#     # print(case_name)
#     body_contour_path = folder+f"/mask_body_contour_{case_name}_Spacing15.nii.gz"
#     # print(body_contour_path)
#     seg_file = nib.load(seg_path)
#     con_file = nib.load(body_contour_path)
#     seg_data = seg_file.get_fdata()
#     con_data = con_file.get_fdata()
#     # print(f"seg shape: {seg_data.shape}, con shape: {con_data.shape}")
#     # seg shape: (467, 467, 730), con shape: (400, 400, 730)

#     # here we need to get x and y to be 256 by 256
#     seg_data_256 = seg_data[105:361, 105:361, :]
#     con_data_256 = con_data[48:304, 48:304, :]

# Folder containing the files
folder = "Spacing15"

# Get a sorted list of body contour files
con_file_list = sorted(glob.glob(folder + "/mask_body_contour_*_Spacing15.nii.gz"))

# Process each body contour file and corresponding segmentation file
for con_path in con_file_list:
    # Extract the case name from the file name
    case_name = os.path.basename(con_path)[18:23]
    
    # Construct the segmentation file path
    seg_path = folder + f"/CTACIVV_{case_name}_MAISI_Spacing15.nii.gz"
    
    # Load the body contour and segmentation NIfTI files
    con_file = nib.load(con_path)
    seg_file = nib.load(seg_path)
    
    # Get the data from the NIfTI files
    con_data = con_file.get_fdata()
    seg_data = seg_file.get_fdata()
    
    # Print the original shapes
    print(f"Original con shape: {con_data.shape}, seg shape: {seg_data.shape}")
    
    # Crop the data
    con_data_256 = con_data[72:328, 72:328, :]
    seg_data_256 = seg_data[105:361, 105:361, :]
    
    # Print the new shapes to confirm
    print(f"Cropped con shape: {con_data_256.shape}, seg shape: {seg_data_256.shape}")
    
    # Save the cropped data (optional)
    # Construct new file names for saving
    con_output_path = folder + f"/cropped_con_{case_name}_Spacing15.nii.gz"
    seg_output_path = folder + f"/cropped_seg_{case_name}_Spacing15.nii.gz"
    
    # Update the affine matrices for the cropped data
    con_affine = con_file.affine.copy()
    seg_affine = seg_file.affine.copy()
    
    con_affine[:3, 3] += [72 * con_affine[0, 0], 72 * con_affine[1, 1], 0]
    seg_affine[:3, 3] += [105 * seg_affine[0, 0], 105 * seg_affine[1, 1], 0]
    
    # Save the cropped NIfTI files
    con_cropped_nifti = nib.Nifti1Image(con_data_256, con_affine, con_file.header)
    seg_cropped_nifti = nib.Nifti1Image(seg_data_256, seg_affine, seg_file.header)
    
    nib.save(con_cropped_nifti, con_output_path)
    nib.save(seg_cropped_nifti, seg_output_path)
    
    print(f"Saved cropped con file to {con_output_path}")
    print(f"Saved cropped seg file to {seg_output_path}")

    # Create a new segmentation map
    new_seg_map = np.zeros_like(con_data_256)
    
    # Apply the binary map logic
    new_seg_map[con_data_256 > 0.5] = 200
    
    # Overlap the segmentation labels from seg_data_256
    new_seg_map[seg_data_256 > 0] = seg_data_256[seg_data_256 > 0]
    
    # Save the new segmentation map with the "emb" tag
    emb_output_path = folder + f"/emb_seg_{case_name}_Spacing15.nii.gz"
    emb_nifti = nib.Nifti1Image(new_seg_map, con_affine, con_file.header)
    
    nib.save(emb_nifti, emb_output_path)
    print(f"Saved new segmentation map to {emb_output_path}")