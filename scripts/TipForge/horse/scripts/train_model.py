import pandas as pd
import xgboost as xgb
import shap
import pickle
import os
import optuna
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_V3 = [
    "distance", "track_quality", "temperature",
    "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
    "h_best_speed", "h_speed_ratio",
    "h_total_prize", "h_days_since",
    "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
    "h_points_l5", "h_top3_l3",
    "d_win_rate", "d_top_3_rate", "hd_pair_runs",
]

FEATURES_V4 = FEATURES_V3 + [
    "h_age", "h_sex", "h_gallop_rate", "dist_diff", "t_win_rate", "t_top3_rate"
]

def train_specific_model(df, features, version_label):
    # Time-based split
    train_df = df[df["date"] < "2025-01-01"]
    test_df  = df[df["date"] >= "2025-01-01"]

    available = [f for f in features if f in df.columns]
    X_train, y_train = train_df[available], train_df["win"]
    X_test,  y_test  = test_df[available],  test_df["win"]

    print(f"\n--- Training Model {version_label} ---")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "scale_pos_weight": 8,
            "objective": "binary:logistic",
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        probs = model.predict_proba(X_test)[:, 1]
        return brier_score_loss(y_test, probs)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    best = study.best_params
    base_model = xgb.XGBClassifier(**best, random_state=42)
    
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, calibrated.predict(X_test))
    
    print(f"Result {version_label}: Brier={brier:.4f}, Acc={acc:.4f}")
    
    # Save V4 if it's the target
    if version_label == "V4":
        model_path = r"E:\Data\TSDP\scripts\TipForge\horse\models\horse_model_v4.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(calibrated, f)
        
        inner = calibrated.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner)
        with open(r"E:\Data\TSDP\scripts\TipForge\horse\models\shap_explainer_v4.pkl", "wb") as f:
            pickle.dump(explainer, f)

    return brier, acc

def main():
    csv_path = r"E:\Data\TSDP\scripts\TipForge\horse\data\training_set_v4.csv"
    if not os.path.exists(csv_path):
        print("V4 dataset not found.")
        return

    df = pd.read_csv(csv_path)
    
    b3, a3 = train_specific_model(df, FEATURES_V3, "V3")
    b4, a4 = train_specific_model(df, FEATURES_V4, "V4")
    
    print("\n" + "="*30)
    print("COMPARISON RESULTS")
    print("="*30)
    print(f"Model V3 Brier: {b3:.4f}")
    print(f"Model V4 Brier: {b4:.4f}")
    improvement = (b3 - b4) / b3 * 100
    print(f"Brier Improvement: {improvement:+.2f}%")
    print(f"Accuracy: V3={a3:.2%}, V4={a4:.2%}")

if __name__ == "__main__":
    main()
