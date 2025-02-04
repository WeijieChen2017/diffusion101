# here we need to load every file in the given folder
# then we call Totalsegmentator to seg them
# example command:
# TotalSegmentator -i CTACIVV_E4055_256.nii.gz -o E4055_256_seg_ml --ml --statistics

# load all files in the target folder
import os

target_folder = "CTACIVV"
seg_result_folder = "CTACIVV_TSeg"
if not os.path.exists(seg_result_folder):
    os.makedirs(seg_result_folder)

file_list = os.listdir(target_folder)
print(f"We find {len(file_list)} files in the target folder")
for file in file_list:
    print(f"--->{file}")

num_files = len(file_list)
# call TotalSegmentator to segment them (TS has been installed in the system)
# not run the commands, just print it
for i, file in enumerate(file_list):
    # command = f"[{i+1}]/[{num_files}] TotalSegmentator -i {target_folder}/{file} -o {seg_result_folder}/{file} --ml"
    # print(f"Executing command: {command}")
    # os.system(command)

    print(f"TotalSegmentator -i {target_folder}/{file} -o {seg_result_folder}/{file} --ml")

