from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
)
from LDM_utils import prepare_dataset, DataConfig, DataLoaderConfig

def create_test_loader(test_names, config: DataConfig):
    # Create test data paths
    test_path_list = []
    for casename in test_names:
        test_path_list.append({
            "CT": f"LDM_adapter/data/CT/CTAC_{casename}_cropped.nii.gz",
            "sCT": f"LDM_adapter/data/sCT/CTAC_{casename}_TS_MAISI.nii.gz",
        })

    # Create test transforms
    test_transforms = Compose(
        [
            LoadImaged(keys=config.input_modality, image_only=True),
            EnsureChannelFirstd(keys=config.input_modality, channel_dim=-1),
        ]
    )

    # Create test dataset
    test_ds = CacheDataset(
        data=test_path_list,
        transform=test_transforms,
        cache_rate=config.test.cache_rate,
        num_workers=config.test.num_workers_cache,
    )

    # Create test dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=config.test.batch_size,
        shuffle=config.test.shuffle,
        num_workers=config.test.num_workers_loader,
    )

    return test_loader

def main():
    # Create data config
    data_config = DataConfig(
        root_folder="LDM_adapter",
        cross_validation=1,  # Using fold_1
        input_modality=["CT", "sCT"],
        train=DataLoaderConfig(
            batch_size=1,
            shuffle=True,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=1.0
        ),
        val=DataLoaderConfig(
            batch_size=1,
            shuffle=False,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=1.0
        ),
        test=DataLoaderConfig(
            batch_size=1,
            shuffle=False,
            num_workers_loader=4,
            num_workers_cache=4,
            cache_rate=1.0
        )
    )
    
    # Create train and val dataloaders using the prepare_dataset function
    train_loader, val_loader, _ = prepare_dataset("LDM_adapter/folds.json", data_config)
    
    # Create test dataloader separately
    test_names = ['E4055', 'E4069', 'E4079', 'E4094', 'E4105', 'E4120', 'E4130', 'E4139', 
                  'E4058', 'E4073', 'E4081', 'E4096', 'E4106', 'E4124', 'E4131', 'E4061', 
                  'E4074', 'E4084', 'E4098', 'E4114', 'E4125', 'E4134', 'E4066', 'E4077', 
                  'E4091', 'E4099', 'E4115', 'E4128', 'E4137', 'E4068', 'E4078', 'E4092', 
                  'E4103', 'E4118', 'E4129', 'E4138']
    test_loader = create_test_loader(test_names, data_config)
    
    # Get first batch and examine data
    first_batch = next(iter(train_loader))
    ct_image = first_batch["CT"]
    sct_image = first_batch["sCT"]
    
    print("Training data:")
    print(f"CT shape: {ct_image.shape}")
    print(f"CT min value: {ct_image.min().item()}")
    print(f"CT max value: {ct_image.max().item()}")
    print(f"sCT shape: {sct_image.shape}")
    print(f"sCT min value: {sct_image.min().item()}")
    print(f"sCT max value: {sct_image.max().item()}")
    
    # Examine validation data
    val_batch = next(iter(val_loader))
    val_ct = val_batch["CT"]
    val_sct = val_batch["sCT"]
    
    print("\nValidation data:")
    print(f"CT shape: {val_ct.shape}")
    print(f"CT min value: {val_ct.min().item()}")
    print(f"CT max value: {val_ct.max().item()}")
    print(f"sCT shape: {val_sct.shape}")
    print(f"sCT min value: {val_sct.min().item()}")
    print(f"sCT max value: {val_sct.max().item()}")
    
    # Examine test data
    test_batch = next(iter(test_loader))
    test_ct = test_batch["CT"]
    test_sct = test_batch["sCT"]
    
    print("\nTest data:")
    print(f"CT shape: {test_ct.shape}")
    print(f"CT min value: {test_ct.min().item()}")
    print(f"CT max value: {test_ct.max().item()}")
    print(f"sCT shape: {test_sct.shape}")
    print(f"sCT min value: {test_sct.min().item()}")
    print(f"sCT max value: {test_sct.max().item()}")

if __name__ == "__main__":
    main()
