import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import load_data, reindex_density

def debug_rounds():
    df = load_data(76986)
    print(f"Loaded {len(df)} rows.")
    print("Round column type:", df['round'].dtype)
    print("Unique rounds in DF:", sorted(df['round'].unique()))
    
    min_r = df['round'].min()
    max_r = df['round'].max()
    print(f"Global Round Range: {min_r} to {max_r}")
    
    # Check one specific player
    pid = df['player_id'].iloc[0]
    p_name = df['player_name'].iloc[0]
    group = df[df['player_id'] == pid]
    print(f"\nPlayer: {p_name} (ID: {pid})")
    print("Original rounds:", sorted(group['round'].unique()))
    
    df_dense = reindex_density(df)
    print(f"\nDense DF size: {len(df_dense)}")
    
    p_dense = df_dense[df_dense['player_id'] == pid]
    print(f"Densified rounds for {p_name}:", sorted(p_dense['round'].unique()))

if __name__ == "__main__":
    debug_rounds()
