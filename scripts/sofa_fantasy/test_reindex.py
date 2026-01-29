import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import reindex_density

def test_reindex():
    data = {
        'player_id': [1, 1, 1, 2, 2],
        'player_name': ['A', 'A', 'A', 'B', 'B'],
        'position': ['M', 'M', 'M', 'F', 'F'],
        'team_id': [10, 10, 10, 20, 20],
        'round': [1, 5, 23, 22, 23],
        'minutes': [90, 90, 90, 45, 90]
    }
    df = pd.DataFrame(data)
    print("Input DF size:", len(df))
    print("Rounds range:", df['round'].min(), "-", df['round'].max())
    
    df_dense = reindex_density(df)
    print("Output DF size:", len(df_dense))
    print("Target Play Counts:\n", (df_dense['minutes'] > 0).value_counts())
    
    print("\nSample for Player B (Gaps 1-21 should be filled):")
    print(df_dense[df_dense['player_id'] == 2].sort_values('round').head(5))
    print("...")
    print(df_dense[df_dense['player_id'] == 2].sort_values('round').tail(5))

if __name__ == "__main__":
    test_reindex()
