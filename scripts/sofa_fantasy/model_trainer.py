import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fantasy_model.json')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

def train_model():
    print("Loading Training Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    
    # FILTER: "Pure Points Model" - Train only on matches where player played.
    # The Start Model handles the 0s.
    print(f"Original Data Size: {len(df)}")
    df = df[df['minutes'] > 0]
    print(f"Filtered Data Size (Played > 0 mins): {len(df)}")

    # 1. Encoding Categoricals (Position) - One Hot
    df = pd.get_dummies(df, columns=['position'], prefix='pos')
    
    # Ensure all position cols exist (in case G, D, M, F missing in snippet)
    for p in ['pos_G', 'pos_D', 'pos_M', 'pos_F']:
        if p not in df.columns:
            df[p] = 0
            
    # Drop non-feature cols and LEAKY cols (current match stats)
    # outcomes we are trying to predict (or components of it) must not be features.
    
    # Explicit Feature List (Safest)
    feature_cols = [
        # Context
        'round', 'is_home', 
        # Lagged Per-Match Stats
        'last_total_points', 'last_minutes', 
        'last_goals', 'last_assists', 'last_rating',
        # Rolling Stats
        'avg_total_points_last_3', 'avg_total_points_last_5', 'avg_total_points_last_38',
        'avg_minutes_last_3', 'avg_minutes_last_5', 'avg_minutes_last_38',
        'avg_goals_last_3', 'avg_goals_last_5', 'avg_goals_last_38',
        'avg_assists_last_3', 'avg_assists_last_5', 'avg_assists_last_38',
        'avg_rating_last_3', 'avg_rating_last_5', 'avg_rating_last_38',
        # Advanced
        'ema_points_span_3', 'ema_points_span_5',
        'starts_last_5', 'xFP_weighted', 'form_vs_season',
        'opp_pos_avg_points_allowed'
    ]
    
    # Add One-Hot Position cols
    pos_cols = [c for c in df.columns if c.startswith('pos_')]
    feature_cols.extend(pos_cols)
    
    target = 'total_points'
    
    # Ensure all cols exist (some might be missing in older csv versions if not careful)
    # But prepare_training_data puts them all there.
    existing_features = [c for c in feature_cols if c in df.columns]
    
    if len(existing_features) < len(feature_cols):
        missing = set(feature_cols) - set(existing_features)
        print(f"Warning: Missing features in CSV: {missing}")
    
    print(f"Features ({len(existing_features)}): {existing_features}")
    
    X = df[existing_features]
    y = df[target]
    rounds = df['round']
    
    # Time Series Cross Validation
    # We split by 'round' index mostly, or just use standard TimeSeriesSplit
    # Since data is sorted (presumably, but let's ensure), we can use TimeSeriesSplit
    
    # Sort by Round/Date for correct splitting
    sort_idx = df.sort_values('round').index
    X_sorted = X.loc[sort_idx]
    y_sorted = y.loc[sort_idx]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    maes = []
    rmses = []
    
    print("\n--- Starting Time Series Validation ---")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    fold = 0
    for train_index, test_index in tscv.split(X_sorted):
        fold += 1
        X_train, X_test = X_sorted.iloc[train_index], X_sorted.iloc[test_index]
        y_train, y_test = y_sorted.iloc[train_index], y_sorted.iloc[test_index]
        
        # Fit
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predict
        preds = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        maes.append(mae)
        rmses.append(rmse)
        
        print(f"Fold {fold}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        
    print(f"\nAverage MAE: {np.mean(maes):.4f}")
    print(f"Average RMSE: {np.mean(rmses):.4f}")
    
    
    # Final Training on Full Data
    print("\nTraining Final Model on Full Dataset...")
    
    # Re-instantiate without early stopping to train on all data
    final_model = xgb.XGBRegressor(
        n_estimators=1000, # Or use average best_iteration from CV? Let's stick to 1000 or slightly less to avoid overfitting
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    final_model.fit(X_sorted, y_sorted, verbose=False)
    
    # Save
    final_model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Feature Importance
    print("\nTop 10 Feature Importances:")
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(10):
        print(f"{i+1}. {existing_features[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    train_model()
