import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import joblib
import os

def main():
    data_path = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\processed_data.csv"
    output_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321"
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} rows.")
    
    # Sort chronologically
    df.sort_values('GAME_DATE', inplace=True)
    
    # Identify feature columns
    # Exclude IDs, dates, target, margin, and odds
    exclude_cols = ['game_id', 'GAME_ID', 'GAME_DATE', 'home_win', 'home_margin', 'odds_home', 'odds_away']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for modeling.")
    
    # Chronological Split (Train 80%, Test 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df['home_win']
    
    X_test = test_df[feature_cols]
    y_test = test_df['home_win']
    
    print(f"Training set: {len(X_train)} samples. Testing set: {len(X_test)} samples.")
    
    # Train XGBoost Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    # using early_stopping_rounds in fit if possible, else standard fit
    try:
        model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
    except Exception as e:
        print(f"Warning on fit kwargs: {e}, falling back to default fit")
        model.fit(X_train, y_train)
    
    # Predictions
    test_df['pred_prob_home'] = model.predict_proba(X_test)[:, 1]
    test_df['pred_class'] = (test_df['pred_prob_home'] >= 0.5).astype(int)
    
    # Evaluation
    acc = accuracy_score(y_test, test_df['pred_class'])
    ll = log_loss(y_test, test_df['pred_prob_home'])
    bs = brier_score_loss(y_test, test_df['pred_prob_home'])
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    
    # Value Betting Logic
    print("\n--- Betting Simulation ---")
    test_df['implied_prob_home'] = 1 / test_df['odds_home']
    test_df['implied_prob_away'] = 1 / test_df['odds_away']
    
    # We require a 2% edge
    margin = 0.02
    
    test_df['bet_home'] = test_df['pred_prob_home'] > (test_df['implied_prob_home'] + margin)
    test_df['bet_away'] = (1 - test_df['pred_prob_home']) > (test_df['implied_prob_away'] + margin)
    
    total_bets = 0
    total_profit = 0.0
    wins = 0
    
    for _, row in test_df.iterrows():
        # Home Bet
        if row['bet_home']:
            total_bets += 1
            if row['home_win'] == 1:
                profit = row['odds_home'] - 1
                total_profit += profit
                wins += 1
            else:
                total_profit -= 1
                
        # Away Bet
        elif row['bet_away'] and not row['bet_home']: # avoid conflicting bets
            total_bets += 1
            if row['home_win'] == 0:
                profit = row['odds_away'] - 1
                total_profit += profit
                wins += 1
            else:
                total_profit -= 1
                
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    hit_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
    
    print(f"Total Bets Placed: {total_bets} out of {len(test_df)} games")
    print(f"Hit Rate: {hit_rate:.1f}%")
    print(f"Total Profit (Units): {total_profit:.2f} U")
    print(f"ROI: {roi:.2f}%")
    
    # Output Results
    results_path = os.path.join(output_dir, "test_results.csv")
    test_df.to_csv(results_path, index=False)
    print(f"\nSaved test results to {results_path}")
    
    model_path = os.path.join(output_dir, "xgb_nba_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
