import pandas as pd
import numpy as np
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_position(pos_str):
    """Maps detailed FBref positions to 4 main categories."""
    if pd.isna(pos_str): return 'OTHER'
    pos = pos_str.upper()
    if 'GK' in pos: return 'GK'
    if 'DF' in pos or 'CB' in pos or 'LB' in pos or 'RB' in pos or 'WB' in pos: return 'DF'
    if 'MF' in pos or 'CM' in pos or 'DM' in pos or 'AM' in pos or 'LM' in pos or 'RM' in pos: return 'MF'
    if 'FW' in pos or 'ST' in pos or 'LW' in pos or 'RW' in pos: return 'FW'
    return 'OTHER'

def generate_rolling_stats(data_dir="./data", output_file="processed_data.parquet", window=5):
    """
    Computes rolling averages for each player's stats to be used for prediction.
    Shifts data by 1 to prevent leakage.
    """
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.parquet")))
    
    logger.info(f"Loading {len(files)} matches...")
    all_dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(all_dfs).reset_index(drop=True)
    
    # Sort by date
    df['match_date'] = pd.to_datetime(df['game'].str.split(' ').str[0])
    df = df.sort_values(['match_date', 'player'])
    
    # Identify numerical features
    exclude = ['league', 'season', 'game', 'team', 'player', 'pos', 'match_date', 'date', 'time', 'game_id']
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    logger.info(f"Computing rolling averages for {len(features)} features...")
    
    # Group by player and compute rolling mean
    # closed='left' ensures we don't include the current match in the average
    def compute_rolling(group):
        rolled = group[features].rolling(window=window, min_periods=1).mean().shift(1)
        # Keep metadata
        for col in exclude:
            if col in group.columns:
                rolled[col] = group[col]
        # Add mapped position
        rolled['pos_mapped'] = group['pos'].apply(map_position)
        return rolled

    processed_df = df.groupby('player', group_keys=False).apply(compute_rolling)
    
    # Fill NAs in rolling stats with 0 (for the first matches of a player)
    processed_df[features] = processed_df[features].fillna(0)
    
    logger.info(f"Saving processed data to {output_file}")
    processed_df.to_parquet(output_file)
    return processed_df

if __name__ == "__main__":
    generate_rolling_stats()
