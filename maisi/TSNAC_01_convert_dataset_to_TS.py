import os
import shutil
from pathlib import Path

case_name_list = [
    "E4055", "E4058", "E4061", "E4063", "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079", "E4080", "E4081", "E4084",
    "E4087", "E4091", "E4092", "E4094", "E4096",
    "E4097", "E4098", "E4099", "E4102", "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
    'E4242', 'E4275', 'E4298', 'E4313',
    'E4245', 'E4276', 'E4299', 'E4317',
    'E4246', 'E4280', 'E4300', 'E4318',
    'E4247', 'E4282', 'E4301', 'E4324',
    'E4248', 'E4283', 'E4302', 'E4325',
    'E4250', 'E4284', 'E4306', 'E4328',
    'E4252', 'E4288', 'E4307', 'E4332',
    'E4259', 'E4289', 'E4308', 'E4335',
    'E4260', 'E4290', 'E4309', 'E4336',
    'E4261', 'E4292', 'E4310', 'E4337',
    'E4273', 'E4297', 'E4312', 'E4338',
]

def organize_files(source_dir, output_base_dir):
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    for case_name in case_name_list:
        # Create case directory
        case_dir = os.path.join(output_base_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Define source and destination file paths
        file_mappings = {
            f"CTAC_{case_name}_TS.nii.gz": "label.nii.gz",
            f"NAC_{case_name}_256.nii.gz": "ct.nii.gz",
            f"CTAC_{case_name}_cropped.nii.gz": "gt.nii.gz"
        }
        
        # Copy and rename files
        for src_name, dst_name in file_mappings.items():
            src_path = os.path.join(source_dir, src_name)
            dst_path = os.path.join(case_dir, dst_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            else:
                print(f"Warning: Source file not found: {src_path}")

if __name__ == "__main__":
    # Set your source and output directories here
    source_directory = "TS_NAC"  # Replace with your source directory
    output_directory = "TS_NAC"  # Replace with your output directory
    
    organize_files(source_directory, output_directory)