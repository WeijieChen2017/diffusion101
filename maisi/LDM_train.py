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
        cross_validation="fold_1",  # Using the exact key from folds.json
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
    
    # Create train, val, and test dataloaders using the prepare_dataset function
    train_loader, val_loader, test_loader = prepare_dataset("LDM_adapter/LDM_folds_with_test.json", data_config)
    
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
