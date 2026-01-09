import torch
import torch.nn as nn
import torch.nn.functional as F

class RisingBallerTransformer(nn.Module):
    def __init__(self, 
                 num_players, 
                 num_positions, 
                 feature_dim, 
                 d_model=128, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super(RisingBallerTransformer, self).__init__()
        
        # 1. Identity Embedding (Player IDs)
        self.player_embedding = nn.Embedding(num_players, d_model)
        
        # 2. Position Embedding (e.g., GK, DF, MF, FW)
        self.pos_embedding = nn.Embedding(num_positions, d_model)
        
        # 3. Performance Feature Projection
        self.feature_projection = nn.Linear(feature_dim, d_model)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Pooling & Heads
        self.to_latent = nn.Identity()
        
        # MPP Head (Masked Player Prediction) - Predict Player ID
        self.mpp_head = nn.Linear(d_model, num_players)
        
        # NMSP Head (Next Match Statistics Prediction) - e.g. Predict Team xG
        # 22 players * d_model -> pooled -> regression
        self.nmsp_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1) # Predicting one metric (e.g. xG)
        )

    def forward(self, player_ids, position_ids, features, mask=None, task='nmsp'):
        """
        Args:
            player_ids: (Batch, 22)
            position_ids: (Batch, 22)
            features: (Batch, 22, feature_dim)
            mask: (Batch, 22) for attention masking if needed
            task: 'mpp' or 'nmsp'
        """
        # Embeddings
        p_emb = self.player_embedding(player_ids)      # (B, 22, d_model)
        pos_emb = self.pos_embedding(position_ids)     # (B, 22, d_model)
        f_proj = self.feature_projection(features)     # (B, 22, d_model)
        
        # Combine: Identity + Position + Performance
        x = p_emb + pos_emb + f_proj                   # (B, 22, d_model)
        
        # Transformer pass
        x = self.transformer_encoder(x, src_key_padding_mask=mask) # (B, 22, d_model)
        
        if task == 'mpp':
            # Predict Player ID for each slot
            return self.mpp_head(x) # (B, 22, num_players)
        
        elif task == 'nmsp':
            # Global Average Pooling of player representations per match
            # For NMSP, we might want to pool by team (first 11 vs last 11)
            # But let's start with a global pool for the whole match context
            x = x.mean(dim=1) # (B, d_model)
            return self.nmsp_head(x) # (B, 1)
        
        return x

if __name__ == "__main__":
    # Quick test
    model = RisingBallerTransformer(num_players=1000, num_positions=5, feature_dim=50)
    
    dummy_p_ids = torch.randint(0, 1000, (8, 22))
    dummy_pos_ids = torch.randint(0, 5, (8, 22))
    dummy_features = torch.randn(8, 22, 50)
    
    out_nmsp = model(dummy_p_ids, dummy_pos_ids, dummy_features, task='nmsp')
    print(f"NMSP Output Shape: {out_nmsp.shape}")
    
    out_mpp = model(dummy_p_ids, dummy_pos_ids, dummy_features, task='mpp')
    print(f"MPP Output Shape: {out_mpp.shape}")
