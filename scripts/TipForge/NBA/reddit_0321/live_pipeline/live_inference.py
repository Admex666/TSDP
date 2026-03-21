import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import scipy.stats
import os
import sys

# Fixed parameters based on Model Training
SIGMA = 13.4312  # Residual Std Dev from our training evaluation

def generate_live_inference():
    pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
    model_path = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\xgb_nba_reg_model.pkl"
    features_path = os.path.join(pipeline_dir, "staging_2_features.csv")
    
    print("\n[Lépés 3: Model Inference & Fair Odds]")
    
    if not os.path.exists(features_path):
        print("❌ Hiba: Nincs Stage 2 fájl (staging_2_features.csv).")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Hiba: Nem található az ML modell! ({model_path})")
        return
        
    # 1. Load Data & Model
    df = pd.read_csv(features_path)
    model = joblib.load(model_path)
    
    print(f"Modell betöltve: {model_path}")
    print(f"Feldolgozandó mérkőzések: {len(df)}")
    
    # 2. Prepare Features
    # Exclude meta/target columns identically to train_regressor.py
    exclude_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'Home_Abbr', 'Away_Abbr', 'home_odds', 'away_odds']
    
    # Find active feature columns (must be numerical)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    X_live = df[feature_cols]
    
    # Check if column order matches the model
    # XGBoost saves feature_names_in_ if trained with a Pandas DataFrame
    try:
        model_features = model.feature_names_in_
        
        # Fill any missing columns (like missing_starters) with 0
        for col in model_features:
            if col not in X_live.columns:
                X_live[col] = 0
                
        # Align columns explicitly just to be safe
        X_live = X_live[model_features]
    except AttributeError:
        # If standard XGBoost without feature names, we assume the same dataframe slice logic preserves order
        pass
        
    # 3. Predict Constraints
    print("Predikció elindítva...")
    df['pred_home_margin'] = model.predict(X_live)
    
    # 4. Fair Odds Transformation (CDF)
    print(f"Győzelmi valószínűségek számítása normál eloszlás alapján (Sigma = {SIGMA})...")
    df['fair_prob_home'] = scipy.stats.norm.cdf(df['pred_home_margin'] / SIGMA)
    df['fair_prob_away'] = 1 - df['fair_prob_home']
    
    df['fair_odds_home'] = 1 / df['fair_prob_home']
    df['fair_odds_away'] = 1 / df['fair_prob_away']
    
    # Output File
    out_path = os.path.join(pipeline_dir, "staging_3_inference.csv")
    df.to_csv(out_path, index=False)
    
    print("\n=======================================================")
    print("--- STAGE 3 OUTPUT: MANUAL REALITY CHECK (EYE TEST) ---")
    print("=======================================================")
    
    # Print the predictions clearly for validation
    for idx, row in df.iterrows():
        matchup = f"{row['Away_Abbr']} @ {row['Home_Abbr']}"
        margin = row['pred_home_margin']
        
        # Determine favored team based on model Prediction
        if margin > 0:
            favored = row['Home_Abbr']
            spread = f"-{margin:.1f}"
        else:
            favored = row['Away_Abbr']
            spread = f"-{abs(margin):.1f}"
            
        print(f"\n🏀 {matchup}")
        print(f"   Model Prediction: {favored} by {abs(margin):.1f} points")
        print(f"   Fair Probabilities: {row['Home_Abbr']} ({row['fair_prob_home']:.1%}) | {row['Away_Abbr']} ({row['fair_prob_away']:.1%})")
        print(f"   Tippmix Odds:       {row['Home_Abbr']} ({row['home_odds']:.2f})  | {row['Away_Abbr']} ({row['away_odds']:.2f})")
    
    print("\n=======================================================")
    print(f"Betting Card Staging elmentve: {out_path}")

if __name__ == "__main__":
    generate_live_inference()
