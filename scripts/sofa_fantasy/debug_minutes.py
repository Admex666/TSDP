import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import load_data, reindex_density

def debug_minutes():
    df = load_data(76986)
    df_dense = reindex_density(df)
    
    print(f"Dense DF size: {len(df_dense)}")
    print("\nMinutes value counts:")
    print(df_dense['minutes'].value_counts().head(10))
    
    print("\nTarget Played (minutes > 0) counts:")
    print((df_dense['minutes'] > 0).value_counts())
    
    # Check one added row
    pid = df['player_id'].iloc[0]
    p_orig_rounds = set(df[df['player_id'] == pid]['round'])
    p_dense = df_dense[df_dense['player_id'] == pid]
    
    added_rounds = [r for r in p_dense['round'] if r not in p_orig_rounds]
    if added_rounds:
        r = added_rounds[0]
        print(f"\nExample added row for pid {pid}, round {r}:")
        print(p_dense[p_dense['round'] == r])
    else:
        print("\nNo added rounds found for first player?")

if __name__ == "__main__":
    debug_minutes()
