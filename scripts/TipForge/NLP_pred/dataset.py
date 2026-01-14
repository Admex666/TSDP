import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import glob

class RisingBallerDataset(Dataset):
    def __init__(self, data_file="processed_data.parquet", player_vocab=None, pos_map=None, features=None):
        # 1. Load processed data
        self.raw_data = pd.read_parquet(data_file)
        
        # Sort by date
        self.raw_data['match_date'] = pd.to_datetime(self.raw_data['game'].str.split(' ').str[0])
        self.raw_data = self.raw_data.sort_values(['match_date', 'game'])
        
        # 2. Setup Vocabs
        if player_vocab is None:
            self.player_vocab = {name: i for i, name in enumerate(self.raw_data['player'].unique())}
        else:
            self.player_vocab = player_vocab
            
        if pos_map is None:
            # Simple mapping for now
            pos_list = self.raw_data['pos_mapped'].unique()
            self.pos_map = {pos: i for i, pos in enumerate(pos_list)}
        else:
            self.pos_map = pos_map
            
        # 3. Identify Numerical Features
        if features is None:
            # Exclude metadata columns
            exclude = ['league', 'season', 'game', 'team', 'player', 'pos', 'match_date', 'date', 'time', 'game_id']
            self.features = [c for c in self.raw_data.columns if c not in exclude and self.raw_data[c].dtype in [np.float64, np.int64]]
        else:
            self.features = features
            
        # 4. Group by matches
        self.match_groups = self.raw_data.groupby('game')
        self.match_list = list(self.match_groups.groups.keys())
        
        # Sort match list chronologically
        self.match_list = sorted(self.match_list, key=lambda x: x.split(' ')[0])

    def __len__(self):
        return len(self.match_list)

    def __getitem__(self, idx):
        match_id = self.match_list[idx]
        group = self.match_groups.get_group(match_id)
        
        # Sort to ensure consistent order (11 home, 11 away - usually 14-16 players total per team with subs)
        # For simplicity, we'll take top 11 by minutes or just first 11 per team
        home_team = group['team'].iloc[0] # First team is home in FBref schedule usually?
        # Actually FBref doesn't explicitly store home/away in the player stats easily without joining schedule
        # Let's just group by team
        teams = group['team'].unique()
        if len(teams) < 2: # Edge case
            return self.__getitem__((idx + 1) % len(self))
            
        team_a_players = group[group['team'] == teams[0]].head(11)
        team_b_players = group[group['team'] == teams[1]].head(11)
        
        # Combine to 22 players
        selected_players = pd.concat([team_a_players, team_b_players])
        
        # Prepare inputs
        player_ids = torch.tensor([self.player_vocab.get(p, 0) for p in selected_players['player']], dtype=torch.long)
        pos_ids = torch.tensor([self.pos_map.get(p, 0) for p in selected_players['pos_mapped']], dtype=torch.long)
        
        # Performance features (normalized/scaled - for now just raw)
        feat_tensor = torch.tensor(selected_players[self.features].fillna(0).values, dtype=torch.float32)
        
        # Target: Home Goals and Away Goals
        # Home Goals = Home Player Gls + Away Player OG
        # Away Goals = Away Player Gls + Home Player OG
        home_players = group[group['team'] == teams[0]]
        away_players = group[group['team'] == teams[1]]
        
        home_score = home_players['Performance_Gls'].sum() + away_players['Performance_OG'].sum()
        away_score = away_players['Performance_Gls'].sum() + home_players['Performance_OG'].sum()
        
        target = torch.tensor([home_score, away_score], dtype=torch.float32)
        
        return {
            'player_ids': player_ids, # (22)
            'pos_ids': pos_ids,       # (22)
            'features': feat_tensor,  # (22, F)
            'target': target          # (2) - [Home, Away]
        }

if __name__ == "__main__":
    ds = RisingBallerDataset(data_dir="./data")
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    
    batch = next(iter(dl))
    print(f"Batch Player IDs shape: {batch['player_ids'].shape}")
    print(f"Batch Features shape: {batch['features'].shape}")
    print(f"Num Features: {len(ds.features)}")
