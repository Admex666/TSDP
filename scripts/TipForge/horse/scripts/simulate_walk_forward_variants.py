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

# Base V4 features (without market odds)
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

def tune_hyperparameters(train_df, features, use_weights=False):
    """Tunes XGBoost hyperparameters using Stratified 3-Fold Cross-Validation on training data."""
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
            
            if use_weights:
                raw_weights = 1.0 / train_df.iloc[train_idx]['market_odds']
                w_tr = raw_weights / raw_weights.mean()
                model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            else:
                model.fit(X_tr, y_tr, verbose=False)
                
            probs = model.predict_proba(X_val)[:, 1]
            briers.append(brier_score_loss(y_val, probs))
            
        return np.mean(briers)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def select_optimal_betting_parameters(df, features, model_label, use_weights=False):
    """
    Simulates a parameter grid search on the 2024 validation split (2024-07-01 to 2025-01-01)
    to select the best Edge and MaxOdds combination.
    """
    print(f"\n--- Parameter Selection on 2024 Validation for {model_label} ---")
    
    # 2024 splits
    train_2024 = df[df["date"] < "2024-07-01"].copy()
    val_2024 = df[(df["date"] >= "2024-07-01") & (df["date"] < "2025-01-01")].copy()
    
    if len(train_2024) == 0 or len(val_2024) == 0:
        print("Warning: Insufficient 2024 data for validation split. Using defaults.")
        return 0.15, 6.0
        
    # Tune on 2024 training subset
    best_params = tune_hyperparameters(train_2024, features, use_weights=use_weights)
    
    # Fit calibrated classifier on 2024 training subset
    X_train = train_2024[features].fillna(train_2024[features].mean())
    y_train = train_2024["win"]
    X_val = val_2024[features].fillna(train_2024[features].mean())
    
    base_model = xgb.XGBClassifier(**best_params, random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    
    if use_weights:
        raw_weights = 1.0 / train_2024['market_odds']
        weights = raw_weights / raw_weights.mean()
        calibrated.fit(X_train, y_train, sample_weight=weights)
    else:
        calibrated.fit(X_train, y_train)
        
    probs = calibrated.predict_proba(X_val)[:, 1]
    val_2024['ml_prob'] = probs
    
    # Drop rows without odds
    val_2024 = val_2024.dropna(subset=["market_odds", "win", "ml_prob"]).copy()
    
    # Normalize probabilities
    val_2024['prob_norm'] = val_2024.groupby('race_id')['ml_prob'].transform(lambda x: x / x.sum())
    val_2024['fair_odds'] = 1.0 / val_2024['prob_norm']
    
    # Grid search
    stake = 1000
    best_val_pnl = -999999.0
    best_cfg = (0.15, 6.0) # default fallback
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            mask = (
                (val_2024['market_odds'] > val_2024['fair_odds'] * (1 + margin))
                & (val_2024['market_odds'] <= max_odds)
            )
            bets = mask.sum()
            if bets < 5:  # Require at least 5 bets to filter out extreme outliers/noise
                continue
                
            pnl = np.where(mask, np.where(val_2024['win'] == 1, (val_2024['market_odds'] - 1) * stake, -stake), 0).sum()
            
            if pnl > best_val_pnl:
                best_val_pnl = pnl
                best_cfg = (margin, max_odds)
                
    chosen_margin, chosen_max_odds = best_cfg
    print(f"Optimal parameters selected on 2024 validation set: Edge={chosen_margin*100:.0f}%, MaxOdds={chosen_max_odds}")
    return chosen_margin, chosen_max_odds

def run_walk_forward(df, features, model_label, output_summary_path, output_grid_path, val_edge, val_max_odds, use_weights=False):
    """Runs a rolling walk-forward simulation for a specific variant over 2025."""
    print(f"\n==================================================")
    print(f"RUNNING WALK-FORWARD SIMULATION: {model_label}")
    print(f"==================================================")
    
    # Split into pre-2025 and 2025 (walk-forward test)
    initial_train_df = df[df["date"] < "2025-01-01"].copy()
    test_months = sorted(df[df["date"] >= "2025-01-01"]["year_month"].unique())
    
    # Tune model on all pre-2025 data
    print("Tuning hyperparameters on all pre-2025 training data...")
    best_params = tune_hyperparameters(initial_train_df, features, use_weights=use_weights)
    
    # Predictions storage
    df_pred = df.copy()
    df_pred['ml_prob'] = np.nan

    print("\nStarting monthly Walk-Forward simulation through 2025...")
    for month in test_months:
        train_mask = df_pred["date"] < f"{month}-01"
        test_mask = df_pred["year_month"] == month
        
        train_sub = df_pred[train_mask].copy()
        test_sub = df_pred[test_mask].copy()
        
        if len(test_sub) == 0:
            continue
            
        X_train = train_sub[features].fillna(train_sub[features].mean())
        y_train = train_sub["win"]
        X_test = test_sub[features].fillna(train_sub[features].mean())
        
        base_model = xgb.XGBClassifier(**best_params, random_state=42)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        
        if use_weights:
            raw_weights = 1.0 / train_sub['market_odds']
            weights = raw_weights / raw_weights.mean()
            calibrated.fit(X_train, y_train, sample_weight=weights)
        else:
            calibrated.fit(X_train, y_train)
        
        probs = calibrated.predict_proba(X_test)[:, 1]
        df_pred.loc[test_mask, 'ml_prob'] = probs

    # Filter for the test set results (2025)
    test_df = df_pred[df_pred['date'] >= '2025-01-01'].copy()
    test_df = test_df.dropna(subset=["market_odds", "win", "ml_prob"]).copy()
    
    # Per-race probability normalization
    test_df['prob_norm'] = test_df.groupby('race_id')['ml_prob'].transform(lambda x: x / x.sum())
    test_df['fair_odds'] = 1 / test_df['prob_norm']

    print(f"\nCompleted Walk-Forward predictions for {len(test_df)} records across {test_df['race_id'].nunique()} races.")

    # Grid Search on 2025 (for showing the heatmap in dashboard)
    stake = 1000
    grid_results = []
    
    print(f"\n{'Margin':>8} | {'MaxOdds':>8} | {'Bets':>6} | {'P/L':>12} | {'ROI':>8} | {'Hit Rate':>8}")
    print("-" * 65)
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            mask = (
                (test_df['market_odds'] > test_df['fair_odds'] * (1 + margin))
                & (test_df['market_odds'] <= max_odds)
            )
            bets = mask.sum()
            if bets == 0:
                pnl = roi = hit_rate = avg_odds = 0
            else:
                pnl = np.where(mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
                staked = bets * stake
                roi = (pnl / staked * 100) if staked > 0 else 0
                hits = test_df[mask]['win'].sum()
                hit_rate = (hits / bets * 100)
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

    # Save grid results
    pd.DataFrame(grid_results).to_csv(output_grid_path, index=False)
    print(f"Grid results saved to {output_grid_path}")

    # Evaluate the PRE-SELECTED parameters (unbiased ROI)
    val_mask = (
        (test_df['market_odds'] > test_df['fair_odds'] * (1 + val_edge))
        & (test_df['market_odds'] <= val_max_odds)
    )
    unbiased_bets = val_mask.sum()
    
    if unbiased_bets > 0:
        unbiased_pnl = np.where(val_mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
        unbiased_staked = unbiased_bets * stake
        unbiased_roi = (unbiased_pnl / unbiased_staked * 100)
        unbiased_hits = test_df[val_mask]['win'].sum()
        unbiased_hr = (unbiased_hits / unbiased_bets * 100)
        unbiased_ao = test_df[val_mask]['market_odds'].mean()
    else:
        unbiased_pnl = 0
        unbiased_roi = 0.0
        unbiased_hr = 0.0
        unbiased_ao = 0.0

    unbiased_label = f"5. Walk-Forward {model_label.split('.')[-1].strip()} (Honest: edge {val_edge*100:.0f}%, MaxOdds <= {val_max_odds:.0f})"
    print(f"\n*** Unbiased out-of-sample ROI on 2025: {unbiased_label} -> ROI {unbiased_roi:+.2f}% ({unbiased_bets} bets)")
    
    # Save summary row
    summary_df = pd.DataFrame([{
        "strategy": unbiased_label,
        "bets": int(unbiased_bets),
        "staked_ft": int(unbiased_bets * stake),
        "pnl_ft": int(unbiased_pnl),
        "roi_pct": round(unbiased_roi, 2),
        "hit_rate_pct": round(unbiased_hr, 2),
        "avg_odds": round(unbiased_ao, 2)
    }])
    summary_df.to_csv(output_summary_path, index=False)
    print(f"Summary stats saved to {output_summary_path}")

def main():
    base_path = "data"
    features_path = os.path.join(base_path, "training_set_v4.csv")
    odds_path = os.path.join(base_path, "training_set_v2_with_odds.csv")

    if not os.path.exists(features_path) or not os.path.exists(odds_path):
        print("Required datasets are missing.")
        return

    print("Loading and merging base datasets...")
    df_features = pd.read_csv(features_path)
    df_odds = pd.read_csv(odds_path)
    
    # Merge
    df = pd.merge(
        df_features, 
        df_odds[['date', 'horse_id', 'market_odds', 'horse_name']], 
        on=['date', 'horse_id'], 
        how='inner'
    )
    df = df.dropna(subset=['market_odds'])
    df = df[df['market_odds'] > 0.0]
    df = df.sort_values("date").reset_index(drop=True)
    df['year_month'] = df['date'].str[:7]
    
    features = [f for f in FEATURES_V4 if f in df.columns]

    # Model 1: Walk-Forward V4 (Base)
    # A. Select parameters on 2024 validation set
    val_edge_v4, val_max_odds_v4 = select_optimal_betting_parameters(
        df=df,
        features=features,
        model_label="V4 Base",
        use_weights=False
    )
    # B. Run walk-forward simulation over 2025
    run_walk_forward(
        df=df,
        features=features,
        model_label="5a. Walk-Forward V4 (Base)",
        output_summary_path=os.path.join(base_path, "walk_forward_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_grid.csv"),
        val_edge=val_edge_v4,
        val_max_odds=val_max_odds_v4,
        use_weights=False
    )

    # Model 2: Walk-Forward V4.2B (Weighted Training)
    # A. Select parameters on 2024 validation set
    val_edge_v42b, val_max_odds_v42b = select_optimal_betting_parameters(
        df=df,
        features=features,
        model_label="V4.2B Weighted",
        use_weights=True
    )
    # B. Run walk-forward simulation over 2025
    run_walk_forward(
        df=df,
        features=features,
        model_label="5b. Walk-Forward V4.2B (Weighted Train)",
        output_summary_path=os.path.join(base_path, "walk_forward_v42b_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_v42b_grid.csv"),
        val_edge=val_edge_v42b,
        val_max_odds=val_max_odds_v42b,
        use_weights=True
    )

if __name__ == "__main__":
    main()
