import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb

# Ensure local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import load_data, calculate_rolling_features, calculate_opponent_strength

def debug_players(player_names):
    print("1. Loading Data (Season 76986)...")
    df_history = load_data(76986)
    
    if df_history.empty:
        print("No data found for season 76986!")
        return
        
    ROUND_TO_PREDICT = 24
    df_history = df_history[df_history['round'] < ROUND_TO_PREDICT] 
    
    # Identify unique player-team from history
    latest = df_history.sort_values(['player_id', 'round']).groupby('player_id').tail(1)
    
    dummies = []
    for _, p in latest.iterrows():
        dummies.append({
            'player_id': p['player_id'],
            'player_name': p['player_name'],
            'position': p['position'],
            'team_id': p['team_id'],
            'round': ROUND_TO_PREDICT,
            'date': pd.Timestamp.now() + pd.Timedelta(days=1),
            'minutes': np.nan # Crucial
        })
    
    df_dummies = pd.DataFrame(dummies)
    full_df = pd.concat([df_history, df_dummies])
    
    print("2. Calculating Rolling Features...")
    # This calls reindex_density internally
    full_df = calculate_rolling_features(full_df)
    
    df_pred = full_df[full_df['round'] == ROUND_TO_PREDICT]
    
    cols = ['player_name', 'round', 'starts_last_5', 'last_minutes', 'avg_minutes_last_3']
    
    # Load Start Model
    START_MODEL_PATH = os.path.join(current_dir, 'models', 'start_model.json')
    if not os.path.exists(START_MODEL_PATH):
        print("Model not found.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(START_MODEL_PATH)
    
    for name in player_names:
        print(f"\n--- Debug: {name} ---")
        p_data = df_pred[df_pred['player_name'].str.contains(name, case=False, na=False)]
        if p_data.empty:
            print("Player not found in prediction set.")
        else:
            print(p_data[cols].to_string(index=False))
            X = p_data[['starts_last_5', 'last_minutes', 'avg_minutes_last_3', 'round']]
            prob = model.predict_proba(X)[:, 1]
            print(f"Model Start Probability: {prob[0]:.4f}")

if __name__ == "__main__":
    # Pope (Recently active), Salah (Recently returned), 
    # Bottman/Reece James (Likely out or returning)
    debug_players(['Nick Pope', 'Mohamed Salah', 'Sven Botman', 'Reece James'])
