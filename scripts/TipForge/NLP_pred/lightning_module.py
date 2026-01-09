import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model import RisingBallerTransformer

class RisingBallerModule(pl.LightningModule):
    def __init__(self, 
                 num_players, 
                 num_positions, 
                 feature_dim, 
                 lr=1e-4,
                 **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = RisingBallerTransformer(
            num_players=num_players,
            num_positions=num_positions,
            feature_dim=feature_dim,
            **model_kwargs
        )
        self.lr = lr
        self.criterion_nmsp = nn.MSELoss()
        self.criterion_mpp = nn.CrossEntropyLoss()

    def forward(self, x, task='nmsp'):
        return self.model(
            player_ids=x['player_ids'],
            position_ids=x['pos_ids'],
            features=x['features'],
            mask=x.get('mask'),
            task=task
        )

    def training_step(self, batch, batch_idx):
        # Default to NMSP for now
        preds = self(batch, task='nmsp')
        loss = self.criterion_nmsp(preds, batch['target'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch, task='nmsp')
        loss = self.criterion_nmsp(preds, batch['target'])
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    # Test initialization
    module = RisingBallerModule(num_players=5000, num_positions=10, feature_dim=96)
    print("Module initialized successfully.")
