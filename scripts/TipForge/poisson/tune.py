import pandas as pd
import numpy as np
from backtest import run_backtest
import concurrent.futures

def evaluate_params(args):
    decay, threshold, df = args
    results_df, history_df = run_backtest(df, ev_threshold=threshold, time_decay=decay, min_matches=100)
    
    if history_df.empty:
        return (decay, threshold, -100, 0)
    
    # Kelly ROI: (Final Bankroll - Initial) / Number of Bets (scaled for comparison)
    final_kelly = history_df['Bankroll_Kelly'].iloc[-1]
    initial_bankroll = 1000
    total_bets = len(results_df) if not results_df.empty else 0
    
    # We use percentage growth for Kelly
    kelly_growth = (final_kelly - initial_bankroll) / initial_bankroll * 100
    return (decay, threshold, kelly_growth, total_bets)

def tune():
    df = pd.read_csv('data/master_football_data.csv')
    
    # Define search space
    decays = [0.0, 0.001, 0.003, 0.005, 0.007, 0.01]
    thresholds = [0.0, 0.02, 0.05, 0.1, 0.15]
    
    tasks = []
    for d in decays:
        for t in thresholds:
            tasks.append((d, t, df))
            
    print(f"Starting tuning with {len(tasks)} combinations...")
    
    results = []
    # Run in parallel for speed
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(evaluate_params, tasks))
        
    tuning_df = pd.DataFrame(results, columns=['Decay', 'Threshold', 'Kelly_Growth_Pct', 'Bets'])
    tuning_df = tuning_df.sort_values('Kelly_Growth_Pct', ascending=False)
    
    print("\n--- Tuning Results: Kelly Growth (Top 10) ---")
    print(tuning_df.head(10).to_string(index=False))
    
    best = tuning_df.iloc[0]
    print(f"\nBest Params: Decay={best['Decay']}, Threshold={best['Threshold']} -> Kelly Growth: {best['Kelly_Growth_Pct']:.2f}%")
    
    tuning_df.to_csv('tuning_results.csv', index=False)

if __name__ == "__main__":
    tune()
