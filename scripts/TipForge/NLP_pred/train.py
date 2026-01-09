import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import RisingBallerDataset
from lightning_module import RisingBallerModule
import logging

logging.basicConfig(level=logging.INFO)

def train_rising_baller():
    # 1. Load Dataset
    try:
        dataset = RisingBallerDataset(data_file="processed_data.parquet")
    except Exception as e:
        print(f"Error loading data: {e}. Please run data_collector.py and preprocess_rolling.py first.")
        return

    # Split into train/val (Using simple time-series split)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    
    # In time-series, we take the last part for validation
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    
    # 2. Initialize Module
    module = RisingBallerModule(
        num_players=len(dataset.player_vocab) + 1,
        num_positions=len(dataset.pos_map) + 1,
        feature_dim=len(dataset.features),
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # 3. Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True
    )
    
    # 4. Fit
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    train_rising_baller()
