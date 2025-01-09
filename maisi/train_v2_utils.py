import json

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
)

from monai.data import CacheDataset, DataLoader

def create_data_loader(
        data_div_json, 
        cv_index, 
        return_train=True, 
        return_val=True, 
        return_test=True, 
        output_size=(256, 256, 32),
        batchsize=1,
        num_samples=4,
        cache_rate=1.,
        is_inference=False,
    ):
    
    if data_div_json is None:
        data_div_json = "../James_data_v3/cv_list.json"
    with open(data_div_json, "r") as f:
        data_div = json.load(f)
    
    cv = cv_index
    cv_test = cv
    cv_val = (cv+1)%5
    cv_train = [(cv+2)%5, (cv+3)%5, (cv+4)%5]

    train_list = data_div[f"cv{cv_train[0]}"] + data_div[f"cv{cv_train[1]}"] + data_div[f"cv{cv_train[2]}"]
    val_list = data_div[f"cv{cv_val}"]
    test_list = data_div[f"cv{cv_test}"]

    print(f"train_list:", train_list)
    print(f"val_list:", val_list)
    print(f"test_list:", test_list)

    # construct the data path list
    train_path_list = []
    val_path_list = []
    test_path_list = []

    for hashname in train_list:
        train_path_list.append({
            "PET": f"../James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"../James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"../James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })
    
    for hashname in val_list:
        val_path_list.append({
            "PET": f"../James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"../James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"../James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })

    for hashname in test_list:
        test_path_list.append({
            "PET": f"../James_data_v3/TOFNAC_256_norm/TOFNAC_{hashname}_norm.nii.gz",
            "CT": f"../James_data_v3/CTACIVV_256_norm/CTACIVV_{hashname}_norm.nii.gz",
            "BODY": f"../James_data_v3/mask/mask_body_contour_{hashname}.nii.gz",
        })

    # construct the divison 
    data_division_dict = {
        "train": train_path_list,
        "val": val_path_list,
        "test": test_path_list,
    }

    input_modality = ["PET", "CT", "BODY"]
    return_dict = {
        "input_modality": input_modality,
        "data_division_dict": data_division_dict,
    }

    # construct the data loader for train
    if return_train:
        
        if not is_inference:
            train_transforms = Compose(
                [
                    LoadImaged(keys=input_modality, image_only=False),
                    EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
                    RandFlipd(keys=input_modality, prob=0.5),
                    RandRotate90d(keys=input_modality, prob=0.5),
                    RandSpatialCropSamplesd(keys=input_modality, roi_size=output_size, num_samples=num_samples),
                ]
            )
        else:
            train_transforms = Compose(
                [
                    LoadImaged(keys=input_modality, image_only=False),
                    EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
                ]
            )

        train_ds = CacheDataset(
            data=train_path_list,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=4,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batchsize,
            shuffle=True,
            num_workers=4,
        )
        
        return_dict["train_loader"] = train_loader
    
    # construct the data loader for val
    if return_val:
        
        if not is_inference:
            val_transforms = Compose(
                [
                    LoadImaged(keys=input_modality, image_only=False),
                    EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
                    RandSpatialCropSamplesd(keys=input_modality, roi_size=output_size, num_samples=num_samples),
                ]
            )
        else:
            val_transforms = Compose(
                [
                    LoadImaged(keys=input_modality, image_only=False),
                    EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
                ]
            )

        val_ds = CacheDataset(
            data=val_path_list,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batchsize,
            shuffle=False,
            num_workers=4,
        )
        
        return_dict["val_loader"] = val_loader

    # construct the data loader for test
    if return_test:
        
        test_transforms = Compose(
            [
                LoadImaged(keys=input_modality, image_only=False),                
                EnsureChannelFirstd(keys=input_modality, channel_dim='no_channel'),
            ]
        )

        test_ds = CacheDataset(
            data=test_path_list,
            transform=test_transforms,
            cache_rate=cache_rate,
            num_workers=4,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batchsize,
            shuffle=False,
            num_workers=4,
        )
        
        return_dict["test_loader"] = test_loader

    return return_dict
