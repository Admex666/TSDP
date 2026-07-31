import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

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

def tune_classifier_hyperparameters(train_df, features):
    """Tunes XGBoost classifier hyperparameters using Stratified 3-Fold CV on training data."""
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
    study.optimize(objective, n_trials=10)
    return study.best_params

def tune_regressor_hyperparameters(train_df, features):
    """Tunes XGBoost regressor hyperparameters using 3-Fold CV on training data for direct ROI regression."""
    X = train_df[features].fillna(train_df[features].mean())
    y = np.where(train_df["win"] == 1, train_df["market_odds"] - 1.0, -1.0)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
            "random_state": 42,
        }
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        mses = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_val)
            mses.append(mean_squared_error(y_val, preds))
            
        return np.mean(mses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    return study.best_params

# Custom Calibration Logic for V4.3C
class CustomOddsCalibrator:
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = LogisticRegression()

    def fit(self, X, y, market_odds):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(X))
        
        # Out of fold predictions to fit calibrator
        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            fold_model = xgb.XGBClassifier(**self.base_model.get_params())
            fold_model.fit(X_tr, y_tr, verbose=False)
            oof_probs[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        # Logit transformation of raw probs
        oof_probs_clipped = np.clip(oof_probs, 1e-5, 1.0 - 1e-5)
        logit_probs = np.log(oof_probs_clipped / (1.0 - oof_probs_clipped))
        log_odds = np.log(market_odds.values)
        
        # Train secondary Logistic Regression calibrator
        calib_X = pd.DataFrame({
            "logit_prob": logit_probs,
            "log_odds": log_odds
        })
        self.calibrator.fit(calib_X, y)
        
        # Train final base model on all data
        self.base_model.fit(X, y, verbose=False)

    def predict_proba(self, X, market_odds):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        raw_probs_clipped = np.clip(raw_probs, 1e-5, 1.0 - 1e-5)
        logit_probs = np.log(raw_probs_clipped / (1.0 - raw_probs_clipped))
        log_odds = np.log(market_odds.values)
        
        calib_X = pd.DataFrame({
            "logit_prob": logit_probs,
            "log_odds": log_odds
        })
        
        # Output calibrated probability
        calib_probs = self.calibrator.predict_proba(calib_X)[:, 1]
        
        # Form [P(0), P(1)] output matching sklearn
        return np.vstack([1.0 - calib_probs, calib_probs]).T

def run_walk_forward_v43(df, features, variant_label, output_summary_path, output_grid_path):
    """
    Runs walk-forward for the four variants V4.3A-D.
    """
    print(f"\n==================================================")
    print(f"RUNNING WALK-FORWARD SIMULATION: {variant_label}")
    print(f"==================================================")
    
    # 1. 2024 Validation split to select optimal parameters
    train_2024 = df[df["date"] < "2024-07-01"].copy()
    val_2024 = df[(df["date"] >= "2024-07-01") & (df["date"] < "2025-01-01")].copy()
    
    # Hyperparameter tuning
    print("Tuning hyperparameters on 2024 training data...")
    if variant_label == "V4.3D (Direct EV Regression)":
        best_params = tune_regressor_hyperparameters(train_2024, features)
        model = xgb.XGBRegressor(**best_params, random_state=42)
        model.fit(train_2024[features].fillna(train_2024[features].mean()), 
                  np.where(train_2024["win"] == 1, train_2024["market_odds"] - 1.0, -1.0))
        val_preds = model.predict(val_2024[features].fillna(train_2024[features].mean()))
        val_2024["pred_ev"] = val_preds
    else:
        best_params = tune_classifier_hyperparameters(train_2024, features)
        base_clf = xgb.XGBClassifier(**best_params, random_state=42)
        
        if variant_label == "V4.3C (Odds-Specific Calibration)":
            model = CustomOddsCalibrator(base_clf)
            model.fit(train_2024[features].fillna(train_2024[features].mean()), train_2024["win"], train_2024["market_odds"])
            val_probs = model.predict_proba(val_2024[features].fillna(train_2024[features].mean()), val_2024["market_odds"])[:, 1]
        else:
            model = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
            model.fit(train_2024[features].fillna(train_2024[features].mean()), train_2024["win"])
            val_probs = model.predict_proba(val_2024[features].fillna(train_2024[features].mean()))[:, 1]
            
        val_2024["pred_prob"] = val_probs

    # Sweeping validation grids for parameter selection
    best_val_pnl = -999999.0
    best_cfg = (0.15, 6.0)
    stake = 1000
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            if variant_label == "V4.3D (Direct EV Regression)":
                mask = (val_2024["pred_ev"] > margin) & (val_2024["market_odds"] <= max_odds)
            else:
                probs = val_2024["pred_prob"]
                if variant_label == "V4.3B (Bayesian Shrinkage)":
                    w = 1.0 / (1.0 + 0.15 * (val_2024["market_odds"] - 1.0))
                    probs = w * probs + (1.0 - w) * (1.0 / val_2024["market_odds"])
                
                # Normalize probabilities per race
                val_2024["temp_prob"] = probs
                val_2024["prob_norm"] = val_2024.groupby("race_id")["temp_prob"].transform(lambda x: x / x.sum())
                val_2024["fair_odds"] = 1.0 / val_2024["prob_norm"]
                
                if variant_label == "V4.3A (Dynamic Margin)":
                    margin_adj = margin * (val_2024["market_odds"] / 3.0)
                    mask = (val_2024["market_odds"] > val_2024["fair_odds"] * (1.0 + margin_adj)) & (val_2024["market_odds"] <= max_odds)
                else:
                    mask = (val_2024["market_odds"] > val_2024["fair_odds"] * (1.0 + margin)) & (val_2024["market_odds"] <= max_odds)
                    
            bets = mask.sum()
            if bets < 5: continue
            
            pnl = np.where(mask, np.where(val_2024['win'] == 1, (val_2024['market_odds'] - 1.0) * stake, -stake), 0).sum()
            if pnl > best_val_pnl:
                best_val_pnl = pnl
                best_cfg = (margin, max_odds)
                
    val_edge, val_max_odds = best_cfg
    print(f"Validation parameter selection complete. Selected Edge={val_edge*100:.0f}%, MaxOdds={val_max_odds}")

    # 2. Out-of-sample 2025 Walk-Forward Loop
    initial_train_df = df[df["date"] < "2025-01-01"].copy()
    test_months = sorted(df[df["date"] >= "2025-01-01"]["year_month"].unique())
    
    print("Tuning final hyperparameters on pre-2025 data...")
    if variant_label == "V4.3D (Direct EV Regression)":
        best_params = tune_regressor_hyperparameters(initial_train_df, features)
    else:
        best_params = tune_classifier_hyperparameters(initial_train_df, features)
        
    df_pred = df.copy()
    if variant_label == "V4.3D (Direct EV Regression)":
        df_pred["pred_ev"] = np.nan
    else:
        df_pred["pred_prob"] = np.nan

    print("Running monthly walk-forward iterations through 2025...")
    for month in test_months:
        train_mask = df_pred["date"] < f"{month}-01"
        test_mask = df_pred["year_month"] == month
        
        train_sub = df_pred[train_mask].copy()
        test_sub = df_pred[test_mask].copy()
        
        if len(test_sub) == 0: continue
        
        X_train = train_sub[features].fillna(train_sub[features].mean())
        X_test = test_sub[features].fillna(train_sub[features].mean())
        
        if variant_label == "V4.3D (Direct EV Regression)":
            y_train = np.where(train_sub["win"] == 1, train_sub["market_odds"] - 1.0, -1.0)
            model = xgb.XGBRegressor(**best_params, random_state=42)
            model.fit(X_train, y_train, verbose=False)
            df_pred.loc[test_mask, "pred_ev"] = model.predict(X_test)
        else:
            y_train = train_sub["win"]
            base_clf = xgb.XGBClassifier(**best_params, random_state=42)
            if variant_label == "V4.3C (Odds-Specific Calibration)":
                model = CustomOddsCalibrator(base_clf)
                model.fit(X_train, y_train, train_sub["market_odds"])
                df_pred.loc[test_mask, "pred_prob"] = model.predict_proba(X_test, test_sub["market_odds"])[:, 1]
            else:
                model = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
                model.fit(X_train, y_train)
                df_pred.loc[test_mask, "pred_prob"] = model.predict_proba(X_test)[:, 1]

    # Evaluate 2025 results
    test_df = df_pred[df_pred["date"] >= "2025-01-01"].copy()
    test_df = test_df.dropna(subset=["market_odds", "win"]).copy()
    
    # Pre-calculate fair odds / predictions for 2025 grid
    if variant_label != "V4.3D (Direct EV Regression)":
        probs = test_df["pred_prob"]
        if variant_label == "V4.3B (Bayesian Shrinkage)":
            w = 1.0 / (1.0 + 0.15 * (test_df["market_odds"] - 1.0))
            probs = w * probs + (1.0 - w) * (1.0 / test_df["market_odds"])
            
        test_df["temp_prob"] = probs
        test_df["prob_norm"] = test_df.groupby("race_id")["temp_prob"].transform(lambda x: x / x.sum())
        test_df["fair_odds"] = 1.0 / test_df["prob_norm"]

    if variant_label == "V4.3A (Dynamic Margin)":
        test_df.to_csv("data/walk_forward_v43a_predictions.csv", index=False)

    grid_results = []
    print(f"\n{'Margin':>8} | {'MaxOdds':>8} | {'Bets':>6} | {'P/L':>12} | {'ROI':>8} | {'Hit Rate':>8}")
    print("-" * 65)
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            if variant_label == "V4.3D (Direct EV Regression)":
                mask = (test_df["pred_ev"] > margin) & (test_df["market_odds"] <= max_odds)
            else:
                if variant_label == "V4.3A (Dynamic Margin)":
                    margin_adj = margin * (test_df["market_odds"] / 3.0)
                    mask = (test_df["market_odds"] > test_df["fair_odds"] * (1.0 + margin_adj)) & (test_df["market_odds"] <= max_odds)
                else:
                    mask = (test_df["market_odds"] > test_df["fair_odds"] * (1.0 + margin)) & (test_df["market_odds"] <= max_odds)
            
            bets = mask.sum()
            if bets == 0:
                pnl = roi = hit_rate = avg_odds = 0.0
            else:
                pnl = np.where(mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
                staked = bets * stake
                roi = (pnl / staked * 100)
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

    # Save grid
    pd.DataFrame(grid_results).to_csv(output_grid_path, index=False)
    print(f"Grid saved to {output_grid_path}")

    # Evaluate validation-selected parameters honestly out-of-sample
    if variant_label == "V4.3D (Direct EV Regression)":
        honest_mask = (test_df["pred_ev"] > val_edge) & (test_df["market_odds"] <= val_max_odds)
    else:
        if variant_label == "V4.3A (Dynamic Margin)":
            margin_adj = val_edge * (test_df["market_odds"] / 3.0)
            honest_mask = (test_df["market_odds"] > test_df["fair_odds"] * (1.0 + margin_adj)) & (test_df["market_odds"] <= val_max_odds)
        else:
            honest_mask = (test_df["market_odds"] > test_df["fair_odds"] * (1.0 + val_edge)) & (test_df["market_odds"] <= val_max_odds)
            
    honest_bets = honest_mask.sum()
    if honest_bets > 0:
        honest_pnl = np.where(honest_mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
        honest_staked = honest_bets * stake
        honest_roi = (honest_pnl / honest_staked * 100)
        honest_hits = test_df[honest_mask]['win'].sum()
        honest_hr = (honest_hits / honest_bets * 100)
        honest_ao = test_df[honest_mask]['market_odds'].mean()
    else:
        honest_pnl = honest_roi = honest_hr = honest_ao = 0.0

    honest_label = f"5. Walk-Forward {variant_label.split('(')[-1].replace(')', '').strip()} (Honest: edge {val_edge*100:.0f}%, MaxOdds <= {val_max_odds:.0f})"
    print(f"\n*** Honest 2025 out-of-sample performance: {honest_label} -> ROI {honest_roi:+.2f}% ({honest_bets} bets)")
    
    # Save summary row
    summary_df = pd.DataFrame([{
        "strategy": honest_label,
        "bets": int(honest_bets),
        "staked_ft": int(honest_bets * stake),
        "pnl_ft": int(honest_pnl),
        "roi_pct": round(honest_roi, 2),
        "hit_rate_pct": round(honest_hr, 2),
        "avg_odds": round(honest_ao, 2)
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

    print("Loading datasets...")
    df_features = pd.read_csv(features_path)
    df_odds = pd.read_csv(odds_path)
    
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

    # Run all 4.3 variants
    run_walk_forward_v43(
        df=df, features=features,
        variant_label="V4.3A (Dynamic Margin)",
        output_summary_path=os.path.join(base_path, "walk_forward_v43a_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_v43a_grid.csv")
    )
    
    run_walk_forward_v43(
        df=df, features=features,
        variant_label="V4.3B (Bayesian Shrinkage)",
        output_summary_path=os.path.join(base_path, "walk_forward_v43b_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_v43b_grid.csv")
    )
    
    run_walk_forward_v43(
        df=df, features=features,
        variant_label="V4.3C (Odds-Specific Calibration)",
        output_summary_path=os.path.join(base_path, "walk_forward_v43c_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_v43c_grid.csv")
    )
    
    run_walk_forward_v43(
        df=df, features=features,
        variant_label="V4.3D (Direct EV Regression)",
        output_summary_path=os.path.join(base_path, "walk_forward_v43d_summary.csv"),
        output_grid_path=os.path.join(base_path, "walk_forward_v43d_grid.csv")
    )

if __name__ == "__main__":
    main()
