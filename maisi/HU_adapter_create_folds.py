import os
import json
import numpy as np
from sklearn.model_selection import KFold

# List of training cases
train_case_name_list = [
    'E4242', 'E4275', 'E4298', 'E4313',
    'E4245', 'E4276', 'E4299', 'E4317', 'E4246',
    'E4280', 'E4300', 'E4318', 'E4247', 'E4282',
    'E4301', 'E4324', 'E4248', 'E4283', 'E4302',
    'E4325', 'E4250', 'E4284', 'E4306', 'E4328',
    'E4252', 'E4288', 'E4307', 'E4332', 'E4259',
    'E4308', 'E4335', 'E4260', 'E4290', 'E4309',
    'E4336', 'E4261', 'E4292', 'E4310', 'E4337',
    'E4273', 'E4297', 'E4312', 'E4338',
]

# Set random seed for reproducibility
np.random.seed(42)

def create_folds(data_list, n_folds=4):
    """
    Divide the input list into n_folds and return a dictionary
    containing the training and validation sets for each fold.
    """
    # Shuffle the list initially to ensure random distribution
    shuffled_list = np.array(data_list.copy())
    np.random.shuffle(shuffled_list)
    
    # Create KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize dictionary to store fold information
    folds_dict = {}
    
    # Split data into folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(shuffled_list)):
        train_cases = shuffled_list[train_idx].tolist()
        val_cases = shuffled_list[val_idx].tolist()
        
        folds_dict[f"fold_{fold_idx+1}"] = {
            "train": train_cases,
            "val": val_cases
        }
    
    return folds_dict

def main():
    print("Creating 4 folds for cross-validation...")
    # Create 4 folds
    folds = create_folds(train_case_name_list, n_folds=4)

    # Create output directory
    root_dir = "HU_adapter_UNet"
    os.makedirs(root_dir, exist_ok=True)

    # Save to JSON file
    json_path = os.path.join(root_dir, "folds.json")
    with open(json_path, 'w') as f:
        json.dump(folds, f, indent=4)

    print(f"Created 4 folds and saved to {json_path}")

    # Print fold statistics
    for fold_name, fold_data in folds.items():
        print(f"{fold_name}: {len(fold_data['train'])} training cases, {len(fold_data['val'])} validation cases")
    
    return 0

if __name__ == "__main__":
    exit(main()) 