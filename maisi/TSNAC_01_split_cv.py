import json
import os

case_name_list = sorted([
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
])

# Create 5 folds based on index % 5
folds = [[] for _ in range(5)]
for i, case in enumerate(case_name_list):
    fold_idx = i % 5
    folds[fold_idx].append(case)

# Create and save 5 different CV splits
for i in range(5):
    # For each split, create train/val/test sets
    # Train: 3 consecutive folds
    # Val: 1 fold
    # Test: 1 fold
    split_dict = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # Rotate the folds for each split configuration
    train_indices = [(i + j) % 5 for j in range(3)]  # Get 3 consecutive folds
    val_index = (i + 3) % 5
    test_index = (i + 4) % 5
    
    # Populate the splits
    for train_idx in train_indices:
        split_dict["train"].extend(folds[train_idx])
    split_dict["val"].extend(folds[val_index])
    split_dict["test"].extend(folds[test_index])
    
    # Sort the lists for consistency
    split_dict["train"].sort()
    split_dict["val"].sort()
    split_dict["test"].sort()
    
    # Save to JSON file
    output_dir = "./TS_NAC"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_file = os.path.join(output_dir, f"TS_NAC_split_cv{i}.json")
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

