import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import PoissonModel

def run_backtest(df, min_matches=50, initial_bankroll=1000, flat_stake=10, kelly_fraction=0.1, ev_threshold=0.05, time_decay=0.0):
    """
    Simulates betting using walk-forward validation and compares against baselines.
    """
    model = PoissonModel()
    results = []
    bankroll_flat = initial_bankroll
    bankroll_kelly = initial_bankroll
    bankroll_fav = initial_bankroll
    bankroll_home = initial_bankroll
    
    # Track history for plotting
    history = []

    # Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Starting backtest on {len(df)} matches (EV > {ev_threshold}, Decay: {time_decay})...")

    for i in range(min_matches, len(df)):
        train_df = df.iloc[:i]
        match = df.iloc[i]
        
        # Fit model on everything up to today with time decay
        model.fit(train_df, prediction_date=match['Date'], time_decay=time_decay)
        
        prediction = model.predict_match(match['HomeTeam'], match['AwayTeam'])
        if not prediction:
            continue

        # Data for this match
        odds = {'H': match['B365H'], 'D': match['B365D'], 'A': match['B365A']}
        probs = {'H': prediction['home_prob'], 'D': prediction['draw_prob'], 'A': prediction['away_prob']}
        actual = match['FTR']

        # 1. BASELINE: Always Bet on Favorite (Lowest Odds)
        fav_side = None
        min_odds = float('inf')
        for side in ['H', 'D', 'A']:
            if not pd.isna(odds[side]) and odds[side] < min_odds:
                min_odds = odds[side]
                fav_side = side
        
        if fav_side:
            outcome_fav = (odds[fav_side] - 1) * flat_stake if actual == fav_side else -flat_stake
            bankroll_fav += outcome_fav

        # 2. BASELINE: Always Bet on Home
        if not pd.isna(odds['H']):
            outcome_home = (odds['H'] - 1) * flat_stake if actual == 'H' else -flat_stake
            bankroll_home += outcome_home

        # 3. POISSON STRATEGY: Best EV
        best_ev_hda = -1
        best_side_hda = None
        
        for side in ['H', 'D', 'A']:
            if pd.isna(odds[side]): continue
            ev = (probs[side] * odds[side]) - 1
            if ev > best_ev_hda:
                best_ev_hda = ev
                best_side_hda = side

        if best_side_hda and best_ev_hda > ev_threshold:
            q = 1 - probs[best_side_hda]
            b = odds[best_side_hda] - 1
            f_star = (b * probs[best_side_hda] - q) / b
            kelly_stake = max(0, f_star * bankroll_kelly * kelly_fraction)
            
            outcome_flat = (odds[best_side_hda] - 1) * flat_stake if actual == best_side_hda else -flat_stake
            outcome_kelly = (odds[best_side_hda] - 1) * kelly_stake if actual == best_side_hda else -kelly_stake
            
            bankroll_flat += outcome_flat
            bankroll_kelly += outcome_kelly

            results.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Market': 'HDA',
                'Side': best_side_hda,
                'Odds': odds[best_side_hda],
                'Prob': probs[best_side_hda],
                'EV': best_ev_hda,
                'Actual': actual,
                'Profit': outcome_flat,
                'Kelly_Profit': outcome_kelly
            })
            
        history.append({
            'Date': match['Date'],
            'Bankroll_Flat': bankroll_flat,
            'Bankroll_Kelly': bankroll_kelly,
            'Bankroll_Fav': bankroll_fav,
            'Bankroll_Home': bankroll_home
        })

    results_df = pd.DataFrame(results)
    history_df = pd.DataFrame(history)
    
    return results_df, history_df

def plot_results(history_df):
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['Date'], history_df['Bankroll_Flat'], label='Poisson (Flat $10)', linewidth=2)
    plt.plot(history_df['Date'], history_df['Bankroll_Fav'], label='Baseline: Always Favorite', linestyle='--', alpha=0.7)
    plt.plot(history_df['Date'], history_df['Bankroll_Home'], label='Baseline: Always Home', linestyle=':', alpha=0.7)
    plt.axhline(1000, color='red', linestyle='-', alpha=0.3)
    plt.title('Poisson Model vs Baselines: Cumulative Profit')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_results.png')
    print("Results plot saved as backtest_results.png")

if __name__ == "__main__":
    df = pd.read_csv('data/master_football_data.csv')
    results_df, history_df = run_backtest(df, ev_threshold=0.05, time_decay=0.005)
    
    if not results_df.empty:
        total_bets = len(results_df)
        won_bets = len(results_df[results_df['Actual'] == results_df['Side']])
        roi_flat = (results_df['Profit'].sum() / (total_bets * 10)) * 100
        
        # Calculate baselines ROI for the same period
        n_matches = len(history_df)
        roi_fav = (history_df['Bankroll_Fav'].iloc[-1] - 1000) / (n_matches * 10) * 100
        roi_home = (history_df['Bankroll_Home'].iloc[-1] - 1000) / (n_matches * 10) * 100

        print(f"\n--- Backtest Results ---")
        print(f"Total Poisson Bets: {total_bets}")
        print(f"Poisson Win Rate: {won_bets/total_bets:.2%}")
        print(f"ROI (Poisson): {roi_flat:.2f}%")
        print(f"ROI (Always Favorite): {roi_fav:.2f}%")
        print(f"ROI (Always Home): {roi_home:.2f}%")
        print(f"Final Bankroll (Poisson Flat): ${history_df['Bankroll_Flat'].iloc[-1]:.2f}")
        print(f"Final Bankroll (Kelly): ${history_df['Bankroll_Kelly'].iloc[-1]:.2f}")
        
        plot_results(history_df)
    else:
        print("No bets were placed.")
