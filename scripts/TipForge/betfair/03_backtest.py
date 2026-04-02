import os
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"

COMMISSION_RATE = 0.05  # 5% commission on net winnings

def apply_drifter_strategy(df):
    """
    Fade drifters: bet against (LAY) horses where odds increase significantly with low early volume.
    Assuming Laying means we win 1 unit if lose, and we lose (bsp - 1) if win.
    """
    # Thresholds
    odds_increase_threshold = df['odds_change'].quantile(0.75)
    low_early_vol_threshold = df['early_volume'].quantile(0.25)
    
    mask = (df['odds_change'] > max(odds_increase_threshold, 1.0)) & (df['early_volume'] < low_early_vol_threshold)
    bets = df[mask].copy()
    
    # Lay calculation for 1 unit flat stake
    # If horse loses (win_lose == 0), we keep the 1 unit stake.
    # If horse wins (win_lose >= 1), we pay out (bsp - 1) units.
    bets['gross_profit'] = np.where(bets['win_lose'] == 0, 1.0, -(bets['bsp'] - 1.0))
    # Net profit: pay 5% commission only on the positive profit
    bets['net_profit'] = np.where(bets['gross_profit'] > 0, bets['gross_profit'] * (1 - COMMISSION_RATE), bets['gross_profit'])
    
    # Calculate volume-weighting weight
    bets['weight'] = bets['total_pre_volume'] / (bets['total_pre_volume'].mean() + 1e-6)
    
    return bets, "Fade Drifters (Lay)"

def apply_steam_strategy(df):
    """
    Follow steam: bet ON (BACK) horses where odds decrease significantly with high early volume.
    Backing means we win (bsp - 1) units if win, and lose 1 unit if lose.
    """
    odds_decrease_threshold = df['odds_change'].quantile(0.25)
    high_early_vol_threshold = df['early_volume'].quantile(0.75)
    
    mask = (df['odds_change'] < min(odds_decrease_threshold, -0.5)) & (df['early_volume'] > high_early_vol_threshold)
    bets = df[mask].copy()
    
    bets['gross_profit'] = np.where(bets['win_lose'] >= 1, bets['bsp'] - 1.0, -1.0)
    bets['net_profit'] = np.where(bets['gross_profit'] > 0, bets['gross_profit'] * (1 - COMMISSION_RATE), bets['gross_profit'])
    bets['weight'] = bets['total_pre_volume'] / (bets['total_pre_volume'].mean() + 1e-6)
    
    return bets, "Follow Steam (Back)"

def apply_longshot_bias_strategy(df):
    """
    Longshot bias: BACK odds > 15, LAY odds < 3.
    """
    # Wait, the rule says "back horses with odds > 15, lay horses with odds < 3"
    # I'll create bets for both and merge them, or handle as a single strategy.
    
    back_mask = df['bsp'] > 15
    lay_mask = df['bsp'] < 3
    
    mask = back_mask | lay_mask
    bets = df[mask].copy()
    
    # For back items
    bets_back_profit = np.where(bets['win_lose'] >= 1, bets['bsp'] - 1.0, -1.0)
    # For lay items
    bets_lay_profit = np.where(bets['win_lose'] == 0, 1.0, -(bets['bsp'] - 1.0))
    
    # Combine based on condition
    bets['gross_profit'] = np.where(bets['bsp'] > 15, bets_back_profit, bets_lay_profit)
    bets['net_profit'] = np.where(bets['gross_profit'] > 0, bets['gross_profit'] * (1 - COMMISSION_RATE), bets['gross_profit'])
    bets['weight'] = bets['total_pre_volume'] / (bets['total_pre_volume'].mean() + 1e-6)
    
    return bets, "Longshot Bias (Back >15, Lay <3)"

def evaluate_bets(bets, strategy_name):
    num_bets = len(bets)
    if num_bets == 0:
        return {
            'Strategy': strategy_name,
            'Bets': 0,
            'Win Rate %': 0,
            'Gross ROI %': 0,
            'Net ROI %': 0,
            'Avg CLV': 0,
            'VW Net ROI %': 0
        }
        
    # For back bets, 'win' is win_lose >= 1. For lay, 'win' is win_lose == 0.
    # We can infer 'win' simply by profit > 0 since we only want to know how many bets won money.
    wins = (bets['gross_profit'] > 0).sum()
    win_rate = wins / num_bets
    
    gross_pnl = bets['gross_profit'].sum()
    net_pnl = bets['net_profit'].sum()
    
    # Since stake is 1 unit for Back, but for Lay the liability can be (bsp-1),  ROI def:
    # "flat staking (1 unit per bet)" - usually implies 1 unit is the stake for back, and 1 unit is the payout for lay.
    # Total stake is just num_bets.
    total_stake = num_bets
    
    gross_roi = gross_pnl / total_stake
    net_roi = net_pnl / total_stake
    
    vw_net_pnl = (bets['net_profit'] * bets['weight']).sum()
    vw_total_stake = bets['weight'].sum()
    vw_net_roi = vw_net_pnl / vw_total_stake if vw_total_stake > 0 else 0
    
    avg_clv = bets['clv'].mean()
    
    return {
        'Strategy': strategy_name,
        'Bets': num_bets,
        'Win Rate %': round(win_rate * 100, 2),
        'Gross ROI %': round(gross_roi * 100, 2),
        'Net ROI %': round(net_roi * 100, 2),
        'Avg CLV': round(avg_clv, 4),
        'VW Net ROI %': round(vw_net_roi * 100, 2)
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    
    if not os.path.exists(df_path):
        logging.error("Master dataset not found!")
        return
        
    df = pd.read_csv(df_path)
    logging.info(f"Loaded master dataset for backtest: {len(df)} rows.")
    
    strategies = [
        apply_drifter_strategy(df),
        apply_steam_strategy(df),
        apply_longshot_bias_strategy(df)
    ]
    
    results = []
    all_bets_dfs = []
    
    for bets, name in strategies:
        metrics = evaluate_bets(bets, name)
        results.append(metrics)
        bets['strategy'] = name
        all_bets_dfs.append(bets)
        logging.info(f"Evaluated {name} - Bets: {metrics['Bets']}, Net ROI: {metrics['Net ROI %']}%")
        
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved backtest summary to {results_path}")

if __name__ == "__main__":
    main()
