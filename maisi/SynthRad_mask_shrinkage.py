import nibabel as nib
import glob
import os

from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.morphology import disk

work_dir = "."
mr_list = sorted(glob.glob(f"{work_dir}/*_mask.nii.gz"))
for path in mr_list:
    nii_file = nib.load(path)
    nii_data = nii_file.get_fdata()
    print(f"Processing {path} with shape {nii_data.shape}")

    # here we shrink the body contour by a circular structuring element with radius 5
    body_contour_data = nii_data.copy()
    len_z = body_contour_data.shape[2]
    structuring_element = disk(5)  # Create a circular structuring element with radius 5
    for z in range(len_z):
        body_contour_data[:, :, z] = binary_erosion(body_contour_data[:, :, z], structure=structuring_element)
    
    body_contour_nii = nib.Nifti1Image(body_contour_data, nii_file.affine, nii_file.header)
    body_contour_path = path.replace("_mask.nii.gz", "_mask_shrink5.nii.gz")
    nib.save(body_contour_nii, body_contour_path)
    print(f"Saved body contour to {body_contour_path}")