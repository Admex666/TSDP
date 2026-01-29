import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.metrics import mean_absolute_error

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')
START_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'start_model.json')
POINTS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fantasy_model.json')

def evaluate_experiment():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing (Same as training)
    df = pd.get_dummies(df, columns=['position'], prefix='pos')
    
    # Features (Must match what was used in training)
    # Ideally should share this list, but hardcoding for now to ensure consistency with model_trainer.py
    # START MODEL FEATURES
    start_features = ['starts_last_5', 'last_minutes', 'avg_minutes_last_3', 'round']
    
    # POINTS MODEL FEATURES (The explicit list we made)
    points_features = [
        'round', 'is_home', 
        'last_total_points', 'last_minutes', 
        'last_goals', 'last_assists', 'last_rating',
        'avg_total_points_last_3', 'avg_total_points_last_5', 'avg_total_points_last_38',
        'avg_minutes_last_3', 'avg_minutes_last_5', 'avg_minutes_last_38',
        'avg_goals_last_3', 'avg_goals_last_5', 'avg_goals_last_38',
        'avg_assists_last_3', 'avg_assists_last_5', 'avg_assists_last_38',
        'avg_rating_last_3', 'avg_rating_last_5', 'avg_rating_last_38',
        'ema_points_span_3', 'ema_points_span_5',
        'starts_last_5', 'xFP_weighted', 'form_vs_season',
        'opp_pos_avg_points_allowed'
    ]
    # Add pos cols
    pos_cols = [c for c in df.columns if c.startswith('pos_')]
    points_features.extend(pos_cols)
    
    # Ensure cols exist
    for c in points_features: 
        if c not in df.columns: df[c] = 0
    for c in start_features:
        if c not in df.columns: df[c] = 0
        
    # Split Test Set (Last 20% rounds)
    # We want to evaluate on ALL players (even those who got 0 points), 
    # because that's what the prediction task is.
    
    df = df.sort_values('round')
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test Set Size: {len(test_df)} rows")
    
    # Load Models
    start_model = xgb.XGBClassifier()
    start_model.load_model(START_MODEL_PATH)
    
    points_model = xgb.XGBRegressor()
    points_model.load_model(POINTS_MODEL_PATH)
    
    # Predict Probability of Playing
    print("Predicting Start Probability...")
    X_start = test_df[start_features]
    probs = start_model.predict_proba(X_start)[:, 1] # Probability of Class 1 (Played)
    test_df['prob_play'] = probs
    
    # Predict Points (Conditional on Playing)
    print("Predicting Expected Points...")
    X_points = test_df[points_features]
    raw_preds = points_model.predict(X_points)
    test_df['pred_points_raw'] = raw_preds
    
    # Validation Logic
    # 1. Baseline: Raw prediction (assuming model learned 0s implicitly? 
    #    No, we trained it ONLY on >0. So predicting 'raw' for a bench player is wrong.
    #    But let's see what it predicts.)
    
    # 2. Weighted: Prob * Raw
    test_df['pred_points_weighted'] = test_df['pred_points_raw'] * test_df['prob_play']
    
    # Calculate MAE
    y_true = test_df['total_points']
    
    mae_raw = mean_absolute_error(y_true, test_df['pred_points_raw'])
    mae_weighted = mean_absolute_error(y_true, test_df['pred_points_weighted'])
    
    print("\n--- Results ---")
    print(f"Baseline MAE (Raw Regressor): {mae_raw:.4f}")
    print(f"Weighted MAE (Prob * Regressor): {mae_weighted:.4f}")
    
    improvement = mae_raw - mae_weighted
    print(f"Improvement: {improvement:.4f} ({(improvement/mae_raw)*100:.2f}%)")
    
    # Inspect a few examples
    print("\n--- Examples (Zero Points / Bench) ---")
    zeros = test_df[test_df['total_points'] == 0].sample(5)
    print(zeros[['player_name', 'prob_play', 'pred_points_raw', 'pred_points_weighted', 'total_points']])
    
    print("\n--- Examples (High Points) ---")
    high = test_df[test_df['total_points'] > 8].sample(5)
    print(high[['player_name', 'prob_play', 'pred_points_raw', 'pred_points_weighted', 'total_points']])

if __name__ == "__main__":
    evaluate_experiment()
