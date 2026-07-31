import pandas as pd
import numpy as np

def run_backtest():
    predictions_path = "data/walk_forward_v43a_predictions.csv"
    if not pd.io.common.file_exists(predictions_path):
        print(f"Predictions file not found at {predictions_path}. Run simulate_walk_forward_variants.py first.")
        return

    df = pd.read_csv(predictions_path)
    
    # Calculate number of runners in each race
    df["runners_count"] = df.groupby("race_id")["horse_id"].transform("count")
    
    # Calculate market rank (1 = favorite, 2 = 2nd favorite, etc.)
    df["market_rank"] = df.groupby("race_id")["market_odds"].rank(method="first", ascending=True)
    
    # Constant stake size
    stake = 1000
    
    print("\n" + "="*70)
    print("BACKTEST: BASELINE-RELATIVE ODDS PLAFOND (MaxOdds = K * Runners)")
    print("Formula: market_odds <= K * runners_count")
    print("="*70)
    print(f"{'Margin':>6} | {'K Mul':>6} | {'MaxOdds':>12} | {'Bets':>6} | {'P/L (Ft)':>12} | {'ROI (%)':>8} | {'Hit %':>7} | {'Avg Odds':>8}")
    print("-" * 88)
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for k in [0.6, 0.8, 1.0, 1.2, 1.4, 99.0]:
            # V4.3A Dynamic Margin formula
            margin_adj = margin * (df["market_odds"] / 3.0)
            
            # Filter condition
            mask = (
                (df["market_odds"] > df["fair_odds"] * (1.0 + margin_adj)) & 
                (df["market_odds"] <= k * df["runners_count"])
            )
            
            bets = mask.sum()
            if bets == 0:
                pnl, roi, hit_rate, avg_odds = 0.0, 0.0, 0.0, 0.0
            else:
                pnl = np.where(mask, np.where(df['win'] == 1, (df['market_odds'] - 1) * stake, -stake), 0).sum()
                roi = (pnl / (bets * stake)) * 100
                hits = df[mask]['win'].sum()
                hit_rate = (hits / bets) * 100
                avg_odds = df[mask]['market_odds'].mean()
                
            k_lbl = f"{k:.1f}" if k < 99 else "all"
            odds_range_lbl = f"<= {k:.1f}*N" if k < 99 else "unlimited"
            print(f"{margin*100:>5.0f}%  | {k_lbl:>6} | {odds_range_lbl:>12} | {bets:>6} | {pnl:>+12,.0f} | {roi:>+7.2f}% | {hit_rate:>5.1f}% | {avg_odds:>8.2f}")
            
    print("\n" + "="*70)
    print("BACKTEST: MARKET RANK CAPPING (MaxRank = X)")
    print("Formula: market_rank <= MaxRank")
    print("="*70)
    print(f"{'Margin':>6} | {'MaxRank':>7} | {'Bets':>6} | {'P/L (Ft)':>12} | {'ROI (%)':>8} | {'Hit %':>7} | {'Avg Odds':>8}")
    print("-" * 75)
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_rank in [1, 2, 3, 4, 5, 99]:
            # V4.3A Dynamic Margin formula
            margin_adj = margin * (df["market_odds"] / 3.0)
            
            # Filter condition
            mask = (
                (df["market_odds"] > df["fair_odds"] * (1.0 + margin_adj)) & 
                (df["market_rank"] <= max_rank)
            )
            
            bets = mask.sum()
            if bets == 0:
                pnl, roi, hit_rate, avg_odds = 0.0, 0.0, 0.0, 0.0
            else:
                pnl = np.where(mask, np.where(df['win'] == 1, (df['market_odds'] - 1) * stake, -stake), 0).sum()
                roi = (pnl / (bets * stake)) * 100
                hits = df[mask]['win'].sum()
                hit_rate = (hits / bets) * 100
                avg_odds = df[mask]['market_odds'].mean()
                
            rank_lbl = f"<= {max_rank}" if max_rank < 99 else "all"
            print(f"{margin*100:>5.0f}%  | {rank_lbl:>7} | {bets:>6} | {pnl:>+12,.0f} | {roi:>+7.2f}% | {hit_rate:>5.1f}% | {avg_odds:>8.2f}")

if __name__ == "__main__":
    run_backtest()
