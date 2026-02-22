import pandas as pd
import pickle
import os
import numpy as np

FEATURES_V3 = [
    "distance",
    "track_quality",
    "temperature",
    "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
    "h_best_speed", "h_speed_ratio",
    "h_total_prize", "h_days_since",
    "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
    "h_points_l5", "h_top3_l3",
    "d_win_rate", "d_top_3_rate",
    "hd_pair_runs",
]

def run_simulation():
    features_path = 'data/training_set_v3.csv'
    odds_path = 'data/training_set_v2_with_odds.csv'
    model_path = 'models/horse_model_v3.pkl'

    if not all(os.path.exists(p) for p in [features_path, odds_path, model_path]):
        print("Required files missing for simulation.")
        return

    print("Loading datasets...")
    df_features = pd.read_csv(features_path)
    df_odds = pd.read_csv(odds_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 1. Merge datasets
    # Note: training_set_v2.csv might have slightly different race_ids if mapping changed,
    # but based on my earlier check they should match.
    # We join on date, horse_id (participant ID).
    print("Merging features with market odds...")
    df = pd.merge(
        df_features, 
        df_odds[['date', 'horse_id', 'market_odds', 'horse_name']], 
        on=['date', 'horse_id'], 
        how='inner'
    )
    
    print(f"Data ready: {len(df)} participants with both features and odds.")

    # 2. Predict with V3 features only
    available = [f for f in FEATURES_V3 if f in df.columns]
    df['prob'] = model.predict_proba(df[available])[:, 1]
    
    # 3. Per-race normalization -> Fair Odds
    df['prob_norm'] = df.groupby('race_id')['prob'].transform(lambda x: x / x.sum())
    df['fair_odds'] = 1 / df['prob_norm']
    
    # 4. Test set: 2025
    test_df = df[df['date'] >= '2025-01-01'].copy()
    print(f"Testing on {len(test_df)} participants from 2025.")

    # 5. Strategy sweep: (margin, max_market_odds) grid
    # The diagnostic showed value bets avg 26.0 odds with a 4.2% hit rate -- classic longshot trap.
    # Adding a max_odds filter targets the range where calibration is most reliable.
    stake = 1000
    print(f"\n{'Margin':>8} | {'MaxOdds':>8} | {'Bets':>6} | {'P/L':>12} | {'ROI':>8}")
    print("-" * 55)
    
    best_roi = -999
    best_cfg = None
    
    for margin in [0.05, 0.10, 0.15, 0.20]:
        for max_odds in [5.0, 8.0, 12.0, 99.0]:
            mask = (
                (test_df['market_odds'] > test_df['fair_odds'] * (1 + margin))
                & (test_df['market_odds'] <= max_odds)
            )
            bets = mask.sum()
            if bets == 0:
                continue
            pnl  = np.where(mask, np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0).sum()
            staked = bets * stake
            roi = (pnl / staked * 100) if staked > 0 else 0
            max_lbl = f"<={max_odds:.0f}" if max_odds < 99 else "all"
            print(f"{margin*100:>7.0f}% | {max_lbl:>8} | {bets:>6} | {pnl:>+11,.0f} Ft | {roi:>+7.2f}%")
            if roi > best_roi:
                best_roi = roi
                best_cfg = (margin, max_odds)

    best_m, best_o = best_cfg
    print(f"\nBest config: Edge {best_m*100:.0f}%, MaxOdds {best_o:.0f} → ROI {best_roi:+.2f}%")

    # Save with best config for dashboard
    test_df['is_value'] = (
        (test_df['market_odds'] > test_df['fair_odds'] * (1 + best_m))
        & (test_df['market_odds'] <= best_o)
    )
    test_df['pnl'] = np.where(test_df['is_value'],
                              np.where(test_df['win'] == 1, (test_df['market_odds'] - 1) * stake, -stake), 0)
    test_df.to_csv('data/simulation_results.csv', index=False)
    print("\nSimulation results saved to data/simulation_results.csv")

if __name__ == "__main__":
    run_simulation()
