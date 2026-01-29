import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import load_data, calculate_rolling_features, calculate_opponent_strength

MODEL_PATH = os.path.join(current_dir, 'models', 'start_model.json')

DATA_PATH = os.path.join(current_dir, 'training_data.csv')

def train_start_model():
    print(f"1. Loading Data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Error: training_data.csv not found. Run prepare_training_data.py first.")
        return
    df = pd.read_csv(DATA_PATH)
    
    print("\n   --- Debug: Minutes Distribution (Raw) ---")
    print(df['minutes'].value_counts().head(5))
    print(f"   Minutes Dtype: {df['minutes'].dtype}")
    
    # Target: Played (> 0 minutes)
    df['target_played'] = (df['minutes'] > 0).astype(int)
    
    print("\n   --- Debug: Target Distribution (Raw) ---")
    print(df['target_played'].value_counts())
    
    # Features
    feature_cols = [
        'starts_last_5',       # Recent reliability
        'last_minutes',        # Did they play last game?
        'avg_minutes_last_3',  # Volume
        'round',               # Season phase
    ]
    
    # Drop NaNs created by rolling
    df = df.dropna(subset=feature_cols)
    
    print("\n   --- Debug: Target Distribution (After DropNA) ---")
    print(df['target_played'].value_counts())
    
    print(f"   Data Shape after clean: {df.shape}")
    
    # Sort for TimeSeries Split
    df = df.sort_values('date')
    
    X = df[feature_cols]
    y = df['target_played']
    
    # Split
    # Use last 20% validation
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Train
    print("2. Training XGBClassifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,    # Slightly slower learning
        max_depth=3,           # Shallower trees
        objective='binary:logistic',
        eval_metric='logloss',
        reg_alpha=1.0,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # Feature Importance
    print("\n   Feature Importance:")
    imps = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    print(imps.sort_values('importance', ascending=False))
    
    # Save
    model.save_model(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_start_model()
