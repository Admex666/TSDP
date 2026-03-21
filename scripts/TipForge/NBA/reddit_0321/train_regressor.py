import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss, accuracy_score
import scipy.stats
import joblib
import os

def main():
    data_path = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\processed_data.csv"
    output_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321"
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} rows.")
    
    df.sort_values('GAME_DATE', inplace=True)
    
    exclude_cols = ['game_id', 'GAME_ID', 'GAME_DATE', 'home_win', 'home_margin', 'odds_home', 'odds_away']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for modeling.")
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df['home_margin']
    
    X_test = test_df[feature_cols]
    y_test = test_df['home_margin']
    y_test_win = test_df['home_win'] # For evaluation
    
    # Train XGBoost Regressor Model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Regression Evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mae = mean_absolute_error(y_test, test_preds)
    
    print(f"\n--- Regression Evaluation ---")
    print(f"RMSE: {rmse:.4f} points")
    print(f"MAE: {mae:.4f} points")
    
    # Residual Calculation for CDF Probability Mapping
    residuals = y_train - train_preds
    sigma = np.std(residuals)
    print(f"Residual Std Dev (Sigma): {sigma:.4f}")
    
    # Convert margins to probabilities
    test_df['pred_margin'] = test_preds
    test_df['pred_prob_home'] = scipy.stats.norm.cdf(test_preds / sigma)
    test_df['pred_class'] = (test_df['pred_prob_home'] >= 0.5).astype(int)
    
    acc = accuracy_score(y_test_win, test_df['pred_class'])
    bs = brier_score_loss(y_test_win, test_df['pred_prob_home'])
    
    print(f"\n--- Derived Classification Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Brier Score: {bs:.4f}")
    
    # Value Betting Logic
    print("\n--- Betting Simulation (Regression) ---")
    test_df['implied_prob_home'] = 1 / test_df['odds_home']
    test_df['implied_prob_away'] = 1 / test_df['odds_away']
    
    margin = 0.02
    
    test_df['bet_home'] = test_df['pred_prob_home'] > (test_df['implied_prob_home'] + margin)
    test_df['bet_away'] = (1 - test_df['pred_prob_home']) > (test_df['implied_prob_away'] + margin)
    
    total_bets = 0
    total_profit = 0.0
    wins = 0
    
    for _, row in test_df.iterrows():
        if row['bet_home']:
            total_bets += 1
            if row['home_win'] == 1:
                profit = row['odds_home'] - 1
                total_profit += profit
                wins += 1
            else:
                total_profit -= 1
        elif row['bet_away'] and not row['bet_home']:
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
    results_path = os.path.join(output_dir, "test_results_reg.csv")
    test_df.to_csv(results_path, index=False)
    
    model_path = os.path.join(output_dir, "xgb_nba_reg_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nSaved regression model to {model_path}")

if __name__ == "__main__":
    main()
