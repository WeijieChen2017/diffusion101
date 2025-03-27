import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def collect_results():
    """Collect results from all folds and create a summary."""
    # Path to fold results
    base_dir = "HU_adapter_UNet"
    
    # Check if folds exist
    folds = [f"fold_{i}" for i in range(1, 5)]
    available_folds = [fold for fold in folds if os.path.exists(os.path.join(base_dir, fold))]
    
    if not available_folds:
        print("No fold results found. Make sure training has completed.")
        return
    
    print(f"Found {len(available_folds)} folds: {', '.join(available_folds)}")
    
    # Collect metrics from each fold
    all_fold_metrics = {}
    best_maes = []
    
    for fold in available_folds:
        fold_dir = os.path.join(base_dir, fold)
        
        # Load loss values
        loss_file = os.path.join(fold_dir, "epoch_loss.npy")
        # Updated file name for MAE in HU
        mae_file = os.path.join(fold_dir, "val_mae_hu.npy")
        
        if os.path.exists(loss_file) and os.path.exists(mae_file):
            loss_values = np.load(loss_file)
            mae_values = np.load(mae_file)
            
            # Find best checkpoint
            best_model_path = os.path.join(fold_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location="cpu")
                # Updated key name for MAE in HU
                best_mae = checkpoint.get("mae_hu", min(mae_values) if len(mae_values) > 0 else None)
                best_epoch = checkpoint.get("epoch", None)
                
                if best_mae is not None:
                    best_maes.append(best_mae)
                    print(f"{fold}: Best MAE = {best_mae:.4f} HU at epoch {best_epoch+1 if best_epoch is not None else 'unknown'}")
            
            # Store metrics
            all_fold_metrics[fold] = {
                "loss": loss_values,
                "mae": mae_values
            }
    
    if not all_fold_metrics:
        print("No metrics found in fold directories.")
        return
    
    # Calculate average MAE across folds
    if best_maes:
        mean_mae = np.mean(best_maes)
        std_mae = np.std(best_maes)
        print(f"\nCross-validation results:")
        print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f} HU")
        
        # Save summary to CSV
        summary_df = pd.DataFrame({
            "fold": [fold.replace("fold_", "") for fold in available_folds],
            "best_mae_hu": best_maes
        })
        summary_df.loc[len(summary_df)] = ["mean", mean_mae]
        summary_df.loc[len(summary_df)] = ["std", std_mae]
        
        summary_path = os.path.join(base_dir, "cv_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
    
    # Plot learning curves
    plot_learning_curves(all_fold_metrics, base_dir)

def plot_learning_curves(all_fold_metrics, base_dir):
    """Plot learning curves for all folds."""
    # Set up figure for plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss for each fold
    for fold, metrics in all_fold_metrics.items():
        epochs = np.arange(1, len(metrics["loss"]) + 1)
        ax1.plot(epochs, metrics["loss"], label=fold)
    
    ax1.set_title("Training Loss (scaled by 3000)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L1 Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation MAE for each fold
    for fold, metrics in all_fold_metrics.items():
        # MAE is calculated every val_interval epochs
        val_interval = 5  # Same as in training script
        val_epochs = np.arange(val_interval, val_interval * (len(metrics["mae"]) + 1), val_interval)
        ax2.plot(val_epochs, metrics["mae"], label=fold)
    
    ax2.set_title("Validation MAE (Hounsfield Units)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (HU)")
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "learning_curves.png"), dpi=300)
    plt.close()
    print(f"Learning curves saved to {os.path.join(base_dir, 'learning_curves.png')}")

if __name__ == "__main__":
    try:
        import torch
        collect_results()
    except Exception as e:
        print(f"Error analyzing results: {str(e)}") 