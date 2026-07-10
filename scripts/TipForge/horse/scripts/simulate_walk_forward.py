import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_V4 = [
    "distance", "track_quality", "temperature",
    "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
    "h_best_speed", "h_speed_ratio",
    "h_total_prize", "h_days_since",
    "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
    "h_points_l5", "h_top3_l3",
    "d_win_rate", "d_top_3_rate", "hd_pair_runs",
    "h_age", "h_sex", "h_gallop_rate", "dist_diff", "t_win_rate", "t_top3_rate"
]

def tune_hyperparameters(train_df, features):
    """Tunes XGBoost hyperparameters on pre-2025 data using Stratified 3-Fold Cross-Validation."""
    print("Tuning hyperparameters on pre-2025 training data...")
    X = train_df[features].fillna(train_df[features].mean())
    y = train_df["win"]
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "scale_pos_weight": 8,
            "objective": "binary:logistic",
            "random_state": 42,
            "eval_metric": "logloss",
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        briers = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            briers.append(brier_score_loss(y_val, probs))
            
        return np.mean(briers)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    print(f"Optimal parameters: {study.best_params}")
    return study.best_params

def run_walk_forward():
    base_path = "data"
    features_path = os.path.join(base_path, "training_set_v4.csv")
    odds_path = os.path.join(base_path, "training_set_v2_with_odds.csv")

    if not os.path.exists(features_path) or not os.path.exists(odds_path):
        print("Required datasets are missing. Run prepare_features.py and merge_odds_results.py first.")
        return

    print("Loading and merging datasets...")
    df_features = pd.read_csv(features_path)
    df_odds = pd.read_csv(odds_path)
    
    # Merge features with market odds
    df = pd.merge(
        df_features, 
        df_odds[['date', 'horse_id', 'market_odds', 'horse_name']], 
        on=['date', 'horse_id'], 
        how='inner'
    )
    
    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)
    df['year_month'] = df['date'].str[:7]
    
    features = [f for f in FEATURES_V4 if f in df.columns]
    
    # Split into pre-2025 (initial train) and 2025 (walk-forward test)
    initial_train_df = df[df["date"] < "2025-01-01"].copy()
    test_months = sorted(df[df["date"] >= "2025-01-01"]["year_month"].unique())
    
    if not test_months:
        print("No test data found for 2025 and after.")
        return

    # Tune model on initial training set
    best_params = tune_hyperparameters(initial_train_df, features)
    
    # Predictions storage
    df['ml_prob'] = np.nan

    print("\nStarting monthly Walk-Forward simulation...")
    for month in test_months:
        # Train on all data before the start of the current month
        train_mask = df["date"] < f"{month}-01"
        test_mask = df["year_month"] == month
        
        train_sub = df[train_mask].copy()
        test_sub = df[test_mask].copy()
        
        if len(test_sub) == 0:
            continue
            
        print(f" -> Month {month}: training on {len(train_sub)} records, predicting {len(test_sub)} records...")
        
        X_train = train_sub[features].fillna(train_sub[features].mean())
        y_train = train_sub["win"]
        X_test = test_sub[features].fillna(train_sub[features].mean()) # Use train mean to fill test NaNs
        
        # Train calibrated classifier
        base_model = xgb.XGBClassifier(**best_params, random_state=42)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        
        probs = calibrated.predict_proba(X_test)[:, 1]
        df.loc[test_mask, 'ml_prob'] = probs

    # Filter for the test set results
    test_df = df[df['date'] >= '2025-01-01'].copy()
    test_df = test_df.dropna(subset=["market_odds", "win", "ml_prob"]).copy()
    
    # Per-race probability normalization
    test_df['prob_norm'] = test_df.groupby('race_id')['ml_prob'].transform(lambda x: x / x.sum())
    test_df['fair_odds'] = 1 / test_df['prob_norm']

    print(f"\nCompleted Walk-Forward predictions for {len(test_df)} records across {test_df['race_id'].nunique()} races.")

    # Save detailed predictions
    predictions_path = os.path.join(base_path, "walk_forward_predictions.csv")
    test_df.to_csv(predictions_path, index=False)
    print(f"Walk-forward predictions saved to {predictions_path}")

    # Value Betting Grid Search
    stake = 1000
    grid_results = []
    
    print(f"\n{'Margin':>8} | {'MaxOdds':>8} | {'Bets':>6} | {'P/L':>12} | {'ROI':>8} | {'Hit Rate':>8}")
    print("-" * 65)
    
    best_roi = -999.0
    best_cfg = None
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            mask = (
                (test_df['market_odds'] > test_df['fair_odds'] * (1 + margin))
                & (test_df['market_odds'] <= max_odds)
            )
            bets = mask.sum()
            if bets == 0:
                continue
                
            pnl = np.where(mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
            staked = bets * stake
            roi = (pnl / staked * 100) if staked > 0 else 0
            hits = test_df[mask]['win'].sum()
            hit_rate = (hits / bets * 100) if bets > 0 else 0
            avg_odds = test_df[mask]['market_odds'].mean()
            
            max_lbl = f"<={max_odds:.0f}" if max_odds < 99 else "all"
            print(f"{margin*100:>7.0f}% | {max_lbl:>8} | {bets:>6} | {pnl:>+11,.0f} Ft | {roi:>+7.2f}% | {hit_rate:>7.1f}%")
            
            grid_results.append({
                "margin_pct": margin * 100,
                "max_odds": max_odds,
                "bets": int(bets),
                "pnl": float(pnl),
                "roi": float(roi),
                "hit_rate": float(hit_rate),
                "avg_odds": float(avg_odds)
            })
            
            if roi > best_roi and bets >= 10:  # Require at least 10 bets to avoid noise
                best_roi = roi
                best_cfg = (margin, max_odds, bets, pnl, hit_rate, avg_odds)

    # Save grid results
    pd.DataFrame(grid_results).to_csv(os.path.join(base_path, "walk_forward_grid.csv"), index=False)

    if best_cfg:
        best_m, best_o, best_b, best_p, best_hr, best_ao = best_cfg
        best_label = f"5. Walk-Forward V4 (edge {best_m*100:.0f}%, MaxOdds <={best_o:.0f})"
        print(f"\n*** Best Walk-Forward Config: {best_label} -> ROI {best_roi:+.2f}%")
        
        # Save summary row for Streamlit comparison
        summary_df = pd.DataFrame([{
            "strategy": best_label,
            "bets": int(best_b),
            "staked_ft": int(best_b * stake),
            "pnl_ft": int(best_p),
            "roi_pct": round(best_roi, 2),
            "hit_rate_pct": round(best_hr, 2),
            "avg_odds": round(best_ao, 2)
        }])
        summary_df.to_csv(os.path.join(base_path, "walk_forward_summary.csv"), index=False)
        print("Summary stats saved to data/walk_forward_summary.csv")
        
        # Flag and save simulation results
        test_df['is_value'] = (
            (test_df['market_odds'] > test_df['fair_odds'] * (1 + best_m))
            & (test_df['market_odds'] <= best_o)
        )
        test_df['pnl'] = np.where(test_df['is_value'],
                                  np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0)
        test_df.to_csv(os.path.join(base_path, "walk_forward_simulation.csv"), index=False)
    else:
        print("\nNo optimal configuration found with enough bets.")

if __name__ == "__main__":
    run_walk_forward()
