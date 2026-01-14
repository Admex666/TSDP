import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from dataset import RisingBallerDataset
from lightning_module import RisingBallerModule
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def validate_model(checkpoint_path=None):
    # 1. Load Dataset
    dataset = RisingBallerDataset(data_file="processed_data.parquet")
    
    # Use the same split as in train.py (last 20% for validation/test)
    train_size = int(len(dataset) * 0.8)
    val_indices = list(range(train_size, len(dataset)))
    val_ds = Subset(dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=1) # One by one for detailed analysis

    # 2. Load Model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = RisingBallerModule.load_from_checkpoint(checkpoint_path)
    else:
        print("Error: No valid checkpoint provided for validation.")
        return
    
    model.eval()
    
    actuals = []
    preds = []
    
    print("\nRunning Validation...")
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch, task='nmsp')
            actuals.append(batch['target'].numpy()[0]) # [Home, Away]
            preds.append(out.numpy()[0])           # [Home, Away]
            
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    # 3. Calculate Metrics
    home_mae = mean_absolute_error(actuals[:, 0], preds[:, 0])
    away_mae = mean_absolute_error(actuals[:, 1], preds[:, 1])
    
    home_rmse = np.sqrt(mean_squared_error(actuals[:, 0], preds[:, 0]))
    away_rmse = np.sqrt(mean_squared_error(actuals[:, 1], preds[:, 1]))
    
    home_r2 = r2_score(actuals[:, 0], preds[:, 0])
    away_r2 = r2_score(actuals[:, 1], preds[:, 1])
    
    print("\n" + "="*40)
    print(" REGRESSION VALIDATION RESULTS")
    print("="*40)
    print(f"Metrics      | Home Team | Away Team")
    print(f"-------------|-----------|-----------")
    print(f"MAE          | {home_mae:.4f}    | {away_mae:.4f}")
    print(f"RMSE         | {home_rmse:.4f}    | {away_rmse:.4f}")
    print(f"R2 Score     | {home_r2:.4f}    | {away_r2:.4f}")
    print("="*40)

    # 4. Visualization (Disabled due to environment issues)
    # plt.figure(figsize=(12, 5))
    # ...
    # plt.savefig("validation_results.png")
    # print("\nValidation plots saved to validation_results.png")

if __name__ == "__main__":
    import glob
    ckpts = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        validate_model(latest_ckpt)
    else:
        print("No checkpoints found. Please train the model first.")
