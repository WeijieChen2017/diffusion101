import os
import glob
import nibabel as nib
import numpy as np

work_dir = "NAC_synCT_MAISI"

vanila_overlap_save_folder = f"{work_dir}/vanila_overlap"
PET_observed_range_save_folder = f"{work_dir}/PET_observation"

target_folder = PET_observed_range_save_folder

# find every .nii.gz file in target_folder
nii_file_list = sorted(glob.glob(target_folder+"/*.nii.gz"))

# create a new folder to save the results
results_folder = f"{work_dir}/results"
os.makedirs(results_folder, exist_ok=True)

# load all files in the target folder
for nii_file_path in nii_file_list:
    nii_file = nib.load(nii_file_path)
    nii_data = nii_file.get_fdata()
    print(f"Processing {nii_file_path} with shape {nii_data.shape}")
    
    # x, y are 400 by 400, so there are four 256 by 256 cubes
    # xy1 is the top left corner, xy2 is the top right corner, xy3 is the bottom left corner, xy4 is the bottom right corner
    # xy1 = nii_data[:256, :256, :]
    # xy2 = nii_data[:256, -256:, :]
    # xy3 = nii_data[-256:, :256, :]
    # xy4 = nii_data[-256:, -256:, :]

    # z is long enough, so we can get 256 with 128 overlap, last is from z_last-256 to z_last
    z_len = nii_data.shape[2]
    num_z = (z_len-256)//128
    z_list = [ nii_data[:, :, i*128:i*128+256] for i in range(num_z) ]
    z_list.append(nii_data[:, :, -256:])

    # save the results
    case_name = os.path.basename(nii_file_path)[:-7]
    for i, z in enumerate(z_list):
        # for each z, we take four corners
        xy1 = z[:256, :256]
        xy2 = z[:256, -256:]
        xy3 = z[-256:, :256]
        xy4 = z[-256:, -256:]

        # we need to change affine and header to save the results
        new_header = nii_file.header.copy()
        new_affine = nii_file.affine.copy()
        new_affine[:2, 2] = 0
        new_affine[2, 2] = 1
        new_header.set_data_shape(xy1.shape)
        new_header.set_xyzt_units('mm')
        new_header.set_qform(new_affine, 1)
        new_header.set_sform(new_affine, 1)

        # save the results
        xy1_nii = nib.Nifti1Image(xy1, new_affine, new_header)
        xy2_nii = nib.Nifti1Image(xy2, new_affine, new_header)
        xy3_nii = nib.Nifti1Image(xy3, new_affine, new_header)
        xy4_nii = nib.Nifti1Image(xy4, new_affine, new_header)

        xy1_path = f"{results_folder}/{case_name}_xy1_{i}.nii.gz"
        xy2_path = f"{results_folder}/{case_name}_xy2_{i}.nii.gz"
        xy3_path = f"{results_folder}/{case_name}_xy3_{i}.nii.gz"
        xy4_path = f"{results_folder}/{case_name}_xy4_{i}.nii.gz"

        nib.save(xy1_nii, xy1_path)
        nib.save(xy2_nii, xy2_path)
        nib.save(xy3_nii, xy3_path)
        nib.save(xy4_nii, xy4_path)

    print(f"Saved results for {case_name}")

print("All done!")
