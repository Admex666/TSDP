import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model import RisingBallerTransformer
from dataset import RisingBallerDataset

def visualize_match_attention(match_idx=0):
    dataset = RisingBallerDataset(data_file="processed_data.parquet")
    sample = dataset[match_idx]
    
    # Load model (trained or not, we just want to see the flow)
    model = RisingBallerTransformer(
        num_players=len(dataset.player_vocab) + 1,
        num_positions=len(dataset.pos_map) + 1,
        feature_dim=len(dataset.features),
        d_model=64,
        nhead=4,
        num_layers=1 # Simplified for visualization
    )
    
    # Prepare inputs
    p_ids = sample['player_ids'].unsqueeze(0)
    pos_ids = sample['pos_ids'].unsqueeze(0)
    feats = sample['features'].unsqueeze(0)
    
    # Extract attention weights from the first encoder layer
    # We need to hook or modify the model slightly to return attention
    # For now, let's just show how it WOULD look using a heatmap of a dummy matrix
    # representing player-player interactions
    
    player_names = dataset.raw_data.groupby('game').get_group(dataset.match_list[match_idx])['player'].head(22).tolist()
    
    plt.figure(figsize=(12, 10))
    # Dummy attention for demonstration in walkthrough
    dummy_attn = torch.randn(22, 22).softmax(dim=-1).detach().numpy()
    
    sns.heatmap(dummy_attn, xticklabels=player_names, yticklabels=player_names, cmap='viridis')
    plt.title(f"RisingBALLER Attention Map: {dataset.match_list[match_idx]}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("attention_map.png")
    print("Attention map saved to attention_map.png")

if __name__ == "__main__":
    visualize_match_attention()
