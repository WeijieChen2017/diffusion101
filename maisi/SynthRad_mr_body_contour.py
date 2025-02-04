import nibabel as nib
import glob
import os

from scipy.ndimage import binary_fill_holes

work_dir = "."
mr_list = sorted(glob.glob(f"{work_dir}/*_mr.nii.gz"))
mr_th = 100
for path in mr_list:
    nii_file = nib.load(path)
    nii_data = nii_file.get_fdata()
    print(f"Processing {path} with shape {nii_data.shape}")

    body_contour_mask = nii_data > mr_th
    body_contour_mask = body_contour_mask.astype(int)
    body_contour_nii = nib.Nifti1Image(body_contour_mask, nii_file.affine, nii_file.header)
    body_contour_path = path.replace(".nii.gz", "_con.nii.gz")

    # take it slice by slice and use binary_fill_holes to fill up the mask
    len_z = body_contour_mask.shape[2]
    for z in range(len_z):
        body_contour_mask[:, :, z] = binary_fill_holes(body_contour_mask[:, :, z])

    nib.save(body_contour_nii, body_contour_path)
    print(f"Saved body contour to {body_contour_path}")