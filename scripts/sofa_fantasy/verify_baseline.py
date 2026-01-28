import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), 'training_data.csv')

def verify_baseline():
    print("Loading Data for Baseline Verification...")
    df = pd.read_csv(DATA_PATH)
    
    # Select Verification Set: Last 5 Rounds (34-38) - Same as ML verification
    verification_rounds = [34, 35, 36, 37, 38]
    print(f"Verifying Baseline (Last 5 Avg) on Rounds: {verification_rounds}")
    
    df_verify = df[df['round'].isin(verification_rounds)].copy()
    
    if df_verify.empty:
        print("No data found for verification rounds!")
        return

    target = 'total_points'
    # Baseline prediction is simply the rolling average of last 5 games
    baseline_pred_col = 'avg_total_points_last_5'
    
    # Handle NaNs in baseline (e.g. if player didn't play enough games)
    # The ML model had 0 filled for these. Let's do the same for fair comparison or check constraints.
    # df_verify = df_verify.dropna(subset=[baseline_pred_col]) 
    
    y_actual = df_verify[target]
    y_pred_baseline = df_verify[baseline_pred_col]
    
    # Metrics
    mae = mean_absolute_error(y_actual, y_pred_baseline)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_baseline))
    
    print(f"\nBaseline Metrics (Naive Last 5 Avg):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print(f"\nComparing to ML Model (from previous run):")
    print(f"ML MAE:  ~1.45 (In-sample/Validation)")
    print(f"ML RMSE: ~1.90")

if __name__ == "__main__":
    verify_baseline()
