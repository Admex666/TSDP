import pandas as pd
import xgboost as xgb
import shap
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss

def train_model(csv_path):
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Feature selection
    features = [
        "distance", "horse_win_rate", "horse_avg_km", 
        "horse_runs", "driver_win_rate", "driver_runs"
    ]
    X = df[features]
    y = df["win"]

    # Split by time (assuming df is sorted by date)
    # Train on 2024, Test on 2025
    train_df = df[df["date"] < "2025-01-01"]
    test_df = df[df["date"] >= "2025-01-01"]
    
    X_train, y_train = train_df[features], train_df["win"]
    X_test, y_test = test_df[features], test_df["win"]

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # Generate SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    with open("models/horse_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    
    print("Model and Explainer saved to models/ directory.")

if __name__ == "__main__":
    train_model("data/training_set.csv")
