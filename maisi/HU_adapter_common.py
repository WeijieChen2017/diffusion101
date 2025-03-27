"""
Common utilities for HU_adapter scripts.
This file replaces the previous HU_adapter_UNet.py with a more organized approach.
"""

import os

# Define root directory for output
ROOT_DIR = "HU_adapter_UNet"

# Lists of case names
TRAIN_CASES = [
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

TEST_CASES = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139',
]

# Constants for HU range
HU_MIN = -1024.0
HU_MAX = 1976.0

# Path helper functions
def get_ct_path(case_name):
    """Get the path to a CT image."""
    return f"maisi/NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz"

def get_sct_path(case_name):
    """Get the path to a synthetic CT image."""
    return f"maisi/NAC_CTAC_Spacing15/CTAC_{case_name}_TS_MAISI.nii.gz"

def get_prediction_path(case_name):
    """Get the path to a predicted synthetic CT image."""
    return os.path.join(ROOT_DIR, "predictions", f"CTAC_{case_name}_predicted.nii.gz")

# Create base directories
def create_base_dirs():
    """Create the base directories used by all scripts."""
    os.makedirs(ROOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "predictions"), exist_ok=True)

# HU conversion functions
def normalize_hu_to_zero_one(hu_values):
    """Convert HU values to 0-1 range."""
    return (hu_values - HU_MIN) / (HU_MAX - HU_MIN)

def denormalize_zero_one_to_hu(normalized_values):
    """Convert 0-1 normalized values back to HU range."""
    return normalized_values * (HU_MAX - HU_MIN) + HU_MIN

# Paths for folds and models
def get_folds_path():
    """Get the path to the folds.json file."""
    return os.path.join(ROOT_DIR, "folds.json")

def get_fold_dir(fold_num):
    """Get the directory for a specific fold."""
    return os.path.join(ROOT_DIR, f"fold_{fold_num}")

def get_best_model_path(fold_num):
    """Get the path to the best model for a fold."""
    return os.path.join(get_fold_dir(fold_num), "best_model.pth")

# Initialize the module by creating necessary directories
create_base_dirs() 