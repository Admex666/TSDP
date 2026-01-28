import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fantasy_model.json')

def verify_model():
    print("Loading Data for Verification...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing (Must match Trainer)
    df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
    for p in ['pos_G', 'pos_D', 'pos_M', 'pos_F']:
        if p not in df_encoded.columns:
            df_encoded[p] = 0
            
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    print(f"Loaded Model from {MODEL_PATH}")
    
    # Select Verification Set: Last 5 Rounds (34-38)
    # The model was trained on ALL data... so this is "in-sample" verification if we trained on full.
    # But usually we want to see how it fits. 
    # Ideally we should strictly hold out.
    # However, for now, let's see how well it learned the patterns of the final rounds.
    
    verification_rounds = [34, 35, 36, 37, 38]
    print(f"Verifying on Rounds: {verification_rounds}")
    
    df_verify = df_encoded[df_encoded['round'].isin(verification_rounds)].copy()
    
    if df_verify.empty:
        print("No data found for verification rounds!")
        return

    drop_cols = ['player_id', 'player_name', 'match_id', 'date', 'team_id', 'opponent_team_id', 'total_points']
    features = [c for c in df_verify.columns if c not in drop_cols]
    target = 'total_points'
    
    X_verify = df_verify[features]
    y_actual = df_verify[target]
    
    # Predict
    preds = model.predict(X_verify)
    df_verify['predicted_points'] = preds
    df_verify['diff'] = df_verify['predicted_points'] - df_verify[target]
    
    # Metrics
    mae = mean_absolute_error(y_actual, preds)
    rmse = np.sqrt(mean_squared_error(y_actual, preds))
    
    print(f"\nVerification Metrics (Rounds {verification_rounds}):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Analyze Specific Round (e.g. 38)
    verify_round(df_verify, 38)

def verify_round(df, round_num):
    print(f"\n--- Detailed Analysis used Round {round_num} ---")
    round_df = df[df['round'] == round_num].copy()
    
    # Top 10 Predicted
    print("\nTop 10 Predicted Performers:")
    top_pred = round_df.sort_values('predicted_points', ascending=False).head(10)
    print(top_pred[['player_name', 'predicted_points', 'total_points', 'diff']].to_string(index=False))
    
    # Top 10 Actual
    print("\nTop 10 Actual Performers:")
    top_actual = round_df.sort_values('total_points', ascending=False).head(10)
    print(top_actual[['player_name', 'predicted_points', 'total_points', 'diff']].to_string(index=False))
    
    # Best XI (Simple heuristic: 1 GK, 3-5 D, 3-5 M, 1-3 F. Max 3 per team is a rule but ignoring for now)
    # Let's just pick highest predicted per position
    print("\n--- Predicted Dream Team (Simple) ---")
    # We need to map back encoded positions to strings if we want to sort by pos easily, 
    # but we can rely on encoded cols.
    
    # Recover position column from original df if needed, or deduce from one-hot.
    # Actually, we kept 'player_name' etc which were in drop_cols but are still in df_verify (we only dropped for X)
    # Wait, 'position' col was get_dummies'd, so it's gone from df_verify unless we kept it?
    # correct: get_dummies replaces the column.
    # We need to reconstruct it or join back.
    
    # Quick fix: find which pos_X is 1
    def get_pos(row):
        if row.get('pos_G', 0) == 1: return 'G'
        if row.get('pos_D', 0) == 1: return 'D'
        if row.get('pos_M', 0) == 1: return 'M'
        if row.get('pos_F', 0) == 1: return 'F'
        return '?'
    
    round_df['position_str'] = round_df.apply(get_pos, axis=1)
    
    for pos in ['G', 'D', 'M', 'F']:
        print(f"\nTop {pos}:")
        chunk = round_df[round_df['position_str'] == pos].sort_values('predicted_points', ascending=False).head(5)
        print(chunk[['player_name', 'predicted_points', 'total_points']].to_string(index=False))

if __name__ == "__main__":
    verify_model()
