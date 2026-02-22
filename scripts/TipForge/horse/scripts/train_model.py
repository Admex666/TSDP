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
    "distance",
    "track_quality",
    "temperature",
    "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
    "h_best_speed", "h_speed_ratio",
    "h_total_prize", "h_days_since",
    "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
    "h_points_l5", "h_top3_l3",
    "d_win_rate", "d_top_3_rate",
    "hd_pair_runs",
]

def train_model(csv_path="data/training_set_v3.csv"):
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Time-based split
    train_df = df[df["date"] < "2025-01-01"]
    test_df  = df[df["date"] >= "2025-01-01"]

    # Fallback columns (if an older CSV is used without all V3 cols)
    available = [f for f in FEATURES_V3 if f in df.columns]
    missing = [f for f in FEATURES_V3 if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features (skipping): {missing}")

    X_train, y_train = train_df[available], train_df["win"]
    X_test,  y_test  = test_df[available],  test_df["win"]

    print(f"Training on {len(X_train)} samples | Testing on {len(X_test)} samples")
    print(f"Features used: {len(available)}")

    # ─── Optuna Hyperparameter Search ────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 6, 12),
            "objective": "binary:logistic",
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        probs = model.predict_proba(X_test)[:, 1]
        return brier_score_loss(y_test, probs)  # minimize

    print("\nRunning Optuna hyperparameter search (50 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best = study.best_params
    print(f"Best params: {best}")
    print(f"Best Brier: {study.best_value:.4f}")

    # ─── Train Final XGBoost with Best Params ────────────────────────────────
    base_model = xgb.XGBClassifier(
        **{k: v for k, v in best.items() if k != "n_estimators"},
        n_estimators=best["n_estimators"],
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # ─── Isotonic Calibration ────────────────────────────────────────────────
    print("\nCalibrating probabilities with Isotonic Regression...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # ─── Evaluate ────────────────────────────────────────────────────────────
    probs_cal = calibrated.predict_proba(X_test)[:, 1]
    preds_cal = calibrated.predict(X_test)

    acc   = accuracy_score(y_test, preds_cal)
    brier = brier_score_loss(y_test, probs_cal)
    ll    = log_loss(y_test, probs_cal)

    print(f"\n--- Model V3 (Calibrated XGBoost) ---")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Brier Score: {brier:.4f}  (lower is better)")
    print(f"  Log Loss   : {ll:.4f}")

    # ─── Save Artifacts ──────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open("models/horse_model_v3.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    print("\nCalibrated model saved to models/horse_model_v3.pkl")

    # SHAP explainer on the inner base estimator
    try:
        inner_model = calibrated.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner_model)
        with open("models/shap_explainer_v3.pkl", "wb") as f:
            pickle.dump(explainer, f)
        print("SHAP explainer saved to models/shap_explainer_v3.pkl")
    except Exception as e:
        print(f"SHAP: {e}")

    print("\nDone. Model V3 is ready.")
    return calibrated, available

if __name__ == "__main__":
    train_model()
