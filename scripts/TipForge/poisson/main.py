from data_loader import download_data
from backtest import run_backtest, plot_results
import pandas as pd
import os

def main():
    print("=== Poisson Betting Model Pipeline ===")
    
    # 1. Download/Load Data
    leagues = ['E0', 'E1', 'D1', 'SP1'] # English PL, Ch'ship, Bundesliga, La Liga
    seasons = ['2425', '2324', '2223']
    
    if not os.path.exists('data/master_football_data.csv'):
        print("Fetching data...")
        df = download_data(leagues, seasons)
        df.to_csv('data/master_football_data.csv', index=False)
    else:
        print("Loading existing data...")
        df = pd.read_csv('data/master_football_data.csv')
    
    # 2. Run Backtest
    print("\nRunning backtest on unified dataset...")
    results_df, history_df = run_backtest(df)
    
    # 3. Summary
    if not results_df.empty:
        total_bets = len(results_df)
        won_bets = len(results_df[results_df['Actual'] == results_df['Side']])
        roi_flat = (results_df['Profit'].sum() / (total_bets * 10)) * 100
        
        print(f"\n--- Final Summary ---")
        print(f"Total Matches Analyzed: {len(df)}")
        print(f"Total Bets Placed: {total_bets}")
        print(f"Win Rate: {won_bets/total_bets:.2%}")
        print(f"ROI (Flat): {roi_flat:.2f}%")
        print(f"Final Bankroll (Flat): ${history_df['Bankroll_Flat'].iloc[-1]:.2f}")
        
        plot_results(history_df)
    else:
        print("No bets were placed. Try lowering the EV threshold in backtest.py.")

if __name__ == "__main__":
    main()
