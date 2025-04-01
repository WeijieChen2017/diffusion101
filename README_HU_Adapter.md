# HU Adapter UNet for CT Synthesis

This repository contains scripts for training and evaluating a 3D UNet model for CT to synthetic CT (sCT) image translation using MONAI.

## Overview

The model is a 3D UNet that takes CT images as input and generates synthetic CT images. The training is performed using 4-fold cross-validation, with each fold running on a separate GPU.

## Scripts

- `maisi/HU_adapter_UNet.py` - Contains the case lists for training and testing
- `maisi/HU_adapter_create_folds.py` - Divides the training list into 4 folds and saves to a JSON file
- `maisi/HU_adapter_train_cv.py` - Performs training for a specific fold on a specific GPU
- `maisi/HU_adapter_run_cv.py` - Runs all 4 cross-validation folds in parallel on 4 GPUs
- `maisi/HU_adapter_analyze_cv.py` - Analyzes the results of the cross-validation and generates summary statistics
- `maisi/HU_adapter_inference.py` - Performs inference on test data using a trained model

## Workflow

### 1. Prepare Data

Ensure your data is organized in the expected format:
- CT images: `NAC_CTAC_Spacing15/CTAC_{case_name}_cropped.nii.gz`
- SCT images: `NAC_CTAC_Spacing15/CTAC_{case_name}_TS_MAISI.nii.gz`

### 2. Create Cross-Validation Folds

```bash
python -m maisi.HU_adapter_create_folds
```

This script divides the training cases into 4 folds and saves them to `HU_adapter_UNet/folds.json`.

### 3. Run Cross-Validation Training

To train all 4 folds in parallel on 4 GPUs:

```bash
python -m maisi.HU_adapter_run_cv
```

Alternatively, to train a specific fold on a specific GPU:

```bash
python -m maisi.HU_adapter_train_cv --fold 1 --gpu 0
```

Training logs will be saved to `HU_adapter_UNet/logs/fold{fold}_gpu{gpu}.log`.

### 4. Analyze Cross-Validation Results

```bash
python -m maisi.HU_adapter_analyze_cv
```

This script analyzes the cross-validation results and produces:
- A summary CSV file: `HU_adapter_UNet/cv_summary.csv`
- Learning curves plot: `HU_adapter_UNet/learning_curves.png`

### 5. Run Inference

After training, you can run inference on test data:

```bash
python -m maisi.HU_adapter_inference
```

This will generate predictions for the test set and save them to `HU_adapter_UNet/predictions/`.

## Model Architecture

The 3D UNet model has the following configuration:
- Spatial dimensions: 3
- Input channels: 1
- Output channels: 1
- Feature channels: (16, 32, 64, 128, 256)
- Strides: (2, 2, 2, 2)
- Residual units: 2

## Training Details

- Loss function: L1 Loss
- Optimizer: AdamW with learning rate 1e-4
- Epochs: 300
- Batch size: 2
- Random crop size: 96×96×96
- HU range normalization: -1024 to 1976 → 0 to 1 