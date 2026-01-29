import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_collector import get_matches_for_round
from feature_engineer import load_data, calculate_rolling_features, calculate_opponent_strength
from predict_next_round import get_next_fixtures

ROUND_TO_PREDICT = 24
PLAYER_NAME = "Mohamed Salah"

def inspect_features():
    print("1. Loading Data...")
    df_history = load_data()
    # Exclude future info
    df_history = df_history[df_history['round'] < ROUND_TO_PREDICT]

    print("2. Preparing Dummies...")
    fixtures_df = get_next_fixtures(ROUND_TO_PREDICT)
    
    latest_player_rows = df_history.sort_values('date').groupby('player_id').tail(1)
    team_fixture_map = fixtures_df.set_index('team_id').to_dict('index')
    
    dummies = []
    
    for _, p_row in latest_player_rows.iterrows():
        pid = p_row['player_id']
        tid = p_row['team_id']
        if tid not in team_fixture_map: continue
        fix = team_fixture_map[tid]
        
        # Mock future entry
        d = {
            'player_id': pid,
            'player_name': p_row['player_name'],
            'position': p_row['position'],
            'team_id': tid,
            'match_id': fix['match_id'],
            'round': ROUND_TO_PREDICT,
            'date': pd.Timestamp.now() + pd.Timedelta(days=1),
            'opponent_team_id': fix['opponent_team_id'],
            'is_home': fix['is_home'],
            'total_points': np.nan, 'minutes': np.nan, 'goals': np.nan, 'assists': np.nan, 'rating': np.nan
        }
        dummies.append(d)
        
    full_df = pd.concat([df_history, pd.DataFrame(dummies)], ignore_index=True)
    
    print("3. Calculating Features...")
    full_df = calculate_rolling_features(full_df)
    full_df = calculate_opponent_strength(full_df)
    
    # Filter for Salah in Round 24
    salah = full_df[(full_df['player_name'] == PLAYER_NAME) & (full_df['round'] == ROUND_TO_PREDICT)]
    
    if salah.empty:
        print(f"Player {PLAYER_NAME} not found in prediction rows.")
        return

    print(f"\n--- Features for {PLAYER_NAME} (Round {ROUND_TO_PREDICT}) ---")
    
    # Feature Columns from predict_next_round.py
    feature_cols = [
        'is_home', 
        'last_total_points', 'last_minutes', 
        'avg_total_points_last_3', 'avg_total_points_last_5', 'avg_total_points_last_38',
        'ema_points_span_3', 'ema_points_span_5',
        'starts_last_5', 'xFP_weighted',
        'avg_minutes_last_3', 
        'avg_goals_last_5', 'avg_assists_last_5',
        'avg_rating_last_5',
        'form_vs_season',
        'opp_pos_avg_points_allowed'
    ]
    
    row = salah.iloc[0]
    for col in feature_cols:
        val = row.get(col, "N/A")
        print(f"{col}: {val}")
        
    print("\n--- Context ---")
    print(f"Opponent Team ID: {row['opponent_team_id']}")
    print(f"Team ID: {row['team_id']}")
    print(f"Position: {row['position']}")

if __name__ == "__main__":
    inspect_features()
