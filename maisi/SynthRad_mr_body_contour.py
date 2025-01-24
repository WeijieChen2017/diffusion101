import nibabel as nib
import glob
import os

work_dir = "."
mr_list = sorted(glob.glob(f"{work_dir}/*.nii.gz"))
mr_th = 250
for path in mr_list:
    nii_file = nib.load(path)
    nii_data = nii_file.get_fdata()
    print(f"Processing {path} with shape {nii_data.shape}")

    body_contour_mask = nii_data > mr_th
    body_contour_mask = body_contour_mask.astype(int)
    body_contour_nii = nib.Nifti1Image(body_contour_mask, nii_file.affine, nii_file.header)
    body_contour_path = path.replace(".nii.gz", "_con.nii.gz")
    nib.save(body_contour_nii, body_contour_path)
    print(f"Saved body contour to {body_contour_path}")