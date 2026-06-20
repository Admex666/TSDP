import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

def main():
    data_path = r"e:\Data\TSDP\scripts\WC_imdb\WC_matches_final.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found. Please run prepare_data.py first.")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    
    # 1. Map stage to stakes (ordinal feature)
    stage_stakes_map = {
        "Group stage": 1.0,
        "Round of 16": 2.0,
        "Quarterfinals": 3.0,
        "Match for 3rd place": 3.5,
        "Semifinals": 4.0,
        "Final": 5.0
    }
    df["stage_stakes"] = df["stage"].map(stage_stakes_map)

    # 2. Select features & target
    feature_cols = [
        "total_goals", "goal_difference", "lead_changes", "comeback_win_draw", 
        "late_goals", "time_of_last_goal", "extra_time", "penalty_shootout", 
        "total_shots", "total_shots_on_target", "total_xg", "xg_difference", 
        "big_chances_total", "big_chances_missed", "possession_imbalance", 
        "fouls_total", "corner_kicks", "red_cards", "yellow_cards", 
        "penalties_awarded", "stage_stakes"
    ]
    
    X = df[feature_cols]
    y = df["rating"]

    print(f"Features loaded: {len(feature_cols)} variables.")
    print(f"Dataset shape: {X.shape}")

    # 3. Define models to test
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest (depth=2)": RandomForestRegressor(n_estimators=150, max_depth=2, random_state=42),
        "Random Forest (depth=3)": RandomForestRegressor(n_estimators=150, max_depth=3, random_state=42),
        "XGBoost (depth=2)": xgb.XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.04, random_state=42)
    }

    # 4. Cross-Validation Evaluation (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for model_name, model in models.items():
        cv_maes = []
        cv_rmses = []
        cv_r2s = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # For linear/ridge models we scale features. 
            # For tree-based models we can also use scaled features without impact.
            # Fit model
            if "Ridge" in model_name:
                # Run quick GridSearchCV for alpha within the fold
                grid = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}, cv=3)
                grid.fit(X_train_scaled, y_train)
                fold_model = grid.best_estimator_
            else:
                fold_model = model
                fold_model.fit(X_train_scaled, y_train)
                
            # Predict
            preds = fold_model.predict(X_val_scaled)
            
            # Metrics
            cv_maes.append(mean_absolute_error(y_val, preds))
            cv_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            cv_r2s.append(r2_score(y_val, preds))
            
        results[model_name] = {
            "MAE": np.mean(cv_maes),
            "RMSE": np.mean(cv_rmses),
            "R2": np.mean(cv_r2s)
        }

    # Print results summary
    print("\n=== Model Performance comparison (5-Fold CV) ===")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string(formatters={"MAE": "{:.3f}".format, "RMSE": "{:.3f}".format, "R2": "{:.3f}".format}))

    # 5. Fit the best model on the entire dataset and analyze feature importance
    best_model_name = results_df["MAE"].idxmin()
    print(f"\n---> Best Model selected: {best_model_name}")

    # Scale full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train full model
    if "Linear" in best_model_name or "Ridge" in best_model_name:
        if "Ridge" in best_model_name:
            grid = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}, cv=5)
            grid.fit(X_scaled, y)
            best_model = grid.best_estimator_
            best_alpha = grid.best_params_["alpha"]
            print(f"Optimal Ridge alpha: {best_alpha}")
        else:
            best_model = models[best_model_name]
            best_model.fit(X_scaled, y)
        
        # Coefficients
        importances = best_model.coef_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient/Importance": importances
        }).sort_values(by="Coefficient/Importance", key=abs, ascending=False)
    elif "Random Forest" in best_model_name:
        best_model = models[best_model_name]
        best_model.fit(X_scaled, y)
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient/Importance": importances
        }).sort_values(by="Coefficient/Importance", ascending=False)
    else:  # XGBoost
        best_model = models[best_model_name]
        best_model.fit(X_scaled, y)
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient/Importance": importances
        }).sort_values(by="Coefficient/Importance", ascending=False)

    print("\n--- Feature Importance / Weights (Full Model) ---")
    for idx, row_imp in imp_df.iterrows():
        print(f"  - {row_imp['Feature']:30s}: {row_imp['Coefficient/Importance']:+.4f}")

    # 6. Predict on sample matches to check prediction accuracy
    df["predicted_rating"] = best_model.predict(X_scaled)
    df["error"] = df["rating"] - df["predicted_rating"]
    
    print("\n--- Predictions for Famous Matches ---")
    famous_matches = [
        ("Argentina", "France"),
        ("Cameroon", "Serbia"),
        ("Qatar", "Ecuador"),
        ("England", "Iran"),
        ("Croatia", "Brazil"),
        ("Morocco", "Spain")
    ]
    
    for h, a in famous_matches:
        match_row = df[((df["home_team"] == h) & (df["away_team"] == a)) | ((df["home_team"] == a) & (df["away_team"] == h))]
        if not match_row.empty:
            row_data = match_row.iloc[0]
            print(f"  - {row_data['home_team']} vs {row_data['away_team']}: Actual = {row_data['rating']:.1f}, Predicted = {row_data['predicted_rating']:.2f} (Error: {row_data['error']:+.2f})")

    # Save predictions
    df.to_csv(data_path, index=False)
    print(f"\nFinal dataset with predictions saved back to: {data_path}")

if __name__ == "__main__":
    main()
