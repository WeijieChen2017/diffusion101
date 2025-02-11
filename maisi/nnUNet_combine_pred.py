import nibabel as nib
import numpy as np
import json

task_101 = "nnUNet/Dataset101_OR0/imagesTs_pred/"
task_102 = "nnUNet/Dataset102_VE0/imagesTs_pred/"
task_103 = "nnUNet/Dataset103_CA0/imagesTs_pred/"
task_104 = "nnUNet/Dataset104_MU0/imagesTs_pred/"
task_105 = "nnUNet/Dataset105_RI0/imagesTs_pred/"

Ts_namelist = [
    "E4068", "E4078", "E4092", "E4103", "E4118",
    "E4129", "E4138", "E4247", "E4260", "E4280",
    "E4290", "E4300", "E4308", "E4317", "E4332",
]

# Dictionary to store all results
unique_values_dict = {}

# Load and process each task's predictions
for task_path in [task_101, task_102, task_103, task_104, task_105]:
    task_name = task_path.split('/')[-3]  # Extract task name from path
    unique_values_dict[task_name] = {}
    
    print(f"\nProcessing {task_path}")
    for name in Ts_namelist:
        file_path = f"{task_path}{name}.nii.gz"
        try:
            # Load the NIfTI file
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Get unique values and convert to list (as numpy arrays aren't JSON serializable)
            unique_vals = np.unique(data).tolist()
            unique_values_dict[task_name][name] = unique_vals
            print(f"{name}: Unique values = {unique_vals}")
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            unique_values_dict[task_name][name] = None
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            unique_values_dict[task_name][name] = None

# Save to JSON file
with open('unique_values.json', 'w') as f:
    json.dump(unique_values_dict, f, indent=4)

print("\nResults have been saved to 'unique_values.json'")


