case_name_list = sorted([
    "E4055", "E4058", "E4061", "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079", "E4081", "E4084",
    "E4091", "E4092", "E4094", "E4096",
    "E4098", "E4099", "E4103",
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
    'E4259', 'E4290', 'E4308', 'E4335',
    'E4260', 'E4292', 'E4309', 'E4336',
    'E4261', 'E4297', 'E4310', 'E4337',
    'E4273', 'E4312', 'E4338',
])

import os
import json

test_cases = [
    "E4055", "E4058", "E4061", "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079", "E4081", "E4084",
    "E4091", "E4092", "E4094", "E4096",
    "E4098", "E4099", "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139"
]

# Create train/val/test split
split_dict = {
    "train": [],
    "val": [],
    "test": []
}

# First separate test cases
for case in case_name_list:
    if case in test_cases:
        split_dict["test"].append(case)

# Get remaining cases for train/val split
remaining_cases = [case for case in case_name_list if case not in test_cases]
remaining_cases.sort()  # Sort for consistency

# Calculate split point for 3:1 ratio
train_size = int(len(remaining_cases) * 0.75)  # 75% for training

# Split remaining cases into train and val
split_dict["train"] = remaining_cases[:train_size]
split_dict["val"] = remaining_cases[train_size:]

# Sort all lists for consistency
split_dict["train"].sort()
split_dict["val"].sort()
split_dict["test"].sort()

# Save to JSON file
output_dir = "./TS_NAC"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
output_file = os.path.join(output_dir, "James36.json")
with open(output_file, 'w') as f:
    json.dump(split_dict, f, indent=4)

