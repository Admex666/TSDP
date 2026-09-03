import os
import itertools
import pandas as pd
from backtest_nb1 import run_walkforward_backtest

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
    
    print("=================================================================")
    print("DYNAMIC ELO MODEL EXPERIMENTS & HYPERPARAMETER BACKTEST")
    print("=================================================================")
    
    # 1. Candidate Models specified in user prompt
    preset_models = {
        "Model A (K=20, H=60, Rho=1.00 - No Decay)": {"k_factor": 20.0, "home_advantage": 60.0, "season_regression": 1.00, "draw_param": 0.26},
        "Model B (K=20, H=60, Rho=0.85 - 85% Decay)": {"k_factor": 20.0, "home_advantage": 60.0, "season_regression": 0.85, "draw_param": 0.26},
        "Model C (K=25, H=55, Rho=0.80 - 80% Decay)": {"k_factor": 25.0, "home_advantage": 55.0, "season_regression": 0.80, "draw_param": 0.26},
    }
    
    preset_results = []
    for name, params in preset_models.items():
        metrics, _, _ = run_walkforward_backtest(
            csv_path,
            k_factor=params["k_factor"],
            home_advantage=params["home_advantage"],
            season_regression=params["season_regression"],
            draw_param=params["draw_param"]
        )
        preset_results.append({
            "Model": name,
            "K-Factor": params["k_factor"],
            "Home Adv (H)": params["home_advantage"],
            "Season Reg (Rho)": params["season_regression"],
            "Draw Param": params["draw_param"],
            "Log Loss": round(metrics["log_loss"], 4),
            "Brier Score": round(metrics["brier_score"], 4),
            "Accuracy": f"{metrics['accuracy']*100:.2f}%"
        })
        
    df_presets = pd.DataFrame(preset_results)
    print("\n--- PRESET MODEL COMPARISON ---")
    print(df_presets.to_string(index=False))
    
    # 2. Grid Search Optimization for Best Log Loss
    print("\n\n-----------------------------------------------------------------")
    print("RUNNING GRID SEARCH OVER PARAMETER SPACE...")
    print("-----------------------------------------------------------------")
    
    k_grid = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    h_grid = [20.0, 40.0, 60.0, 80.0, 100.0]
    rho_grid = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    draw_grid = [0.22, 0.25, 0.26, 0.28, 0.30]
    
    best_log_loss = float("inf")
    best_params = None
    best_metrics = None
    best_preds_df = None
    best_season_df = None
    
    search_space = list(itertools.product(k_grid, h_grid, rho_grid, draw_grid))
    print(f"Total Grid Search combinations to test: {len(search_space)}")
    
    for k, h, rho, d in search_space:
        metrics, season_df, preds_df = run_walkforward_backtest(
            csv_path,
            k_factor=k,
            home_advantage=h,
            season_regression=rho,
            draw_param=d
        )
        
        if metrics["log_loss"] < best_log_loss:
            best_log_loss = metrics["log_loss"]
            best_params = {"k_factor": k, "home_advantage": h, "season_regression": rho, "draw_param": d}
            best_metrics = metrics
            best_preds_df = preds_df
            best_season_df = season_df

    print("\n=================================================================")
    print("OPTIMIZED BEST DYNAMIC ELO MODEL FOUND")
    print("=================================================================")
    print(f"Best K-Factor:         {best_params['k_factor']}")
    print(f"Best Home Advantage:   {best_params['home_advantage']} Elo points")
    print(f"Best Season Regression:{best_params['season_regression']*100:.0f}% (Rho={best_params['season_regression']})")
    print(f"Best Draw Parameter:   {best_params['draw_param']}")
    print(f"-----------------------------------------------------------------")
    print(f"Best Log Loss:         {best_metrics['log_loss']:.4f}")
    print(f"Best Brier Score:      {best_metrics['brier_score']:.4f}")
    print(f"Best Accuracy:         {best_metrics['accuracy']*100:.2f}%")
    print("=================================================================")
    
    print("\n--- Season Breakdown for Best Model ---")
    print(best_season_df.to_string(index=False))
    
    # Save predictions log of best model
    preds_out = os.path.join(os.path.dirname(__file__), "nbi_elo_predictions_2015_2025.csv")
    best_preds_df.to_csv(preds_out, index=False, encoding='utf-8-sig')
    print(f"\n[+] Full backtest prediction log saved to: {preds_out}")

if __name__ == "__main__":
    main()
