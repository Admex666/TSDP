"""
Baseline Strategy Comparison
Compares multiple simple strategies against Model V3 on the 2025 test set.

Strategies:
1. Random    — pick a random horse per race, flat bet
2. Favorite  — always bet on the lowest-odds horse per race
3. Form      — always bet on the horse with the highest h_points_l5 per race
4. Speed     — always bet on the horse with the highest h_avg_speed_l5 per race
5. V3 Value  — model edge >= 15%, market odds <= 8 (our best config)
"""

import pandas as pd
import numpy as np
import pickle
import os
import random
random.seed(42)
np.random.seed(42)

FEATURES_V3 = [
    "distance", "track_quality", "temperature",
    "h_win_rate", "h_top_3_rate", "h_avg_percentile", "h_avg_speed",
    "h_best_speed", "h_speed_ratio",
    "h_total_prize", "h_days_since",
    "h_win_rate_l5", "h_top_3_rate_l5", "h_avg_percentile_l5", "h_avg_speed_l5",
    "h_points_l5", "h_top3_l3",
    "d_win_rate", "d_top_3_rate",
    "hd_pair_runs",
]

STAKE = 1000  # Ft per bet


def simulate_strategy(name, bet_mask, df, stake=STAKE):
    """Given a boolean mask of bets on df, calculate metrics."""
    # Only consider rows with valid odds
    valid = df["market_odds"].notna()
    bet_mask = bet_mask & valid
    bets = bet_mask.sum()
    if bets == 0:
        return {"strategy": name, "bets": 0, "staked_ft": 0, "pnl_ft": 0,
                "roi_pct": 0.0, "hit_rate_pct": 0.0, "avg_odds": 0.0}
    raw_pnl = np.where(bet_mask,
                       np.where(df["win"] == 1, (df["market_odds"] - 1) * stake, -stake),
                       0)
    pnl = float(np.nansum(raw_pnl))
    hits = df[bet_mask]["win"].sum()
    roi = pnl / (bets * stake) * 100
    return {
        "strategy": name,
        "bets": int(bets),
        "staked_ft": int(bets * stake),
        "pnl_ft": int(pnl),
        "roi_pct": round(roi, 2),
        "hit_rate_pct": round(float(hits) / bets * 100, 2),
        "avg_odds": round(float(df[bet_mask]["market_odds"].mean()), 2),
    }


def run_baselines():
    # ── Load Data ────────────────────────────────────────────────────────────
    sim_path = "data/simulation_results.csv"
    feat_path = "data/training_set_v3.csv"
    odds_path = "data/training_set_v2_with_odds.csv"
    model_path = "models/horse_model_v3.pkl"

    if not all(os.path.exists(p) for p in [sim_path, feat_path, odds_path, model_path]):
        print("Missing required files. Run simulate_value_betting.py first.")
        return

    print("Loading data...")
    sim_df = pd.read_csv(sim_path)          # V3 model results with market_odds
    feat_df = pd.read_csv(feat_path)        # All V3 features
    odds_df = pd.read_csv(odds_path)        # Original odds table with horse names

    # Merge features + odds for baseline calculations (same merge as simulation)
    base_df = pd.merge(
        feat_df,
        odds_df[["date", "horse_id", "market_odds", "horse_name"]],
        on=["date", "horse_id"],
        how="inner"
    )
    base_df = base_df[base_df["date"] >= "2025-01-01"].copy()
    print(f"Test set (2025): {len(base_df)} participants across {base_df['race_id'].nunique()} races")

    results = []

    # ── 1. Random Strategy ────────────────────────────────────────────────────
    # Pick exactly one random horse per race
    random_picks = set()
    for race_id, group in base_df.groupby("race_id"):
        picked = group.sample(1).index[0]
        random_picks.add(picked)
    mask_random = base_df.index.isin(random_picks)
    results.append(simulate_strategy("1. Random (1 per race)", mask_random, base_df))

    # ── 2. Favorite Strategy ──────────────────────────────────────────────────
    # Bet on the horse with the LOWEST market odds per race (bookmaker favorite)
    fav_picks = set()
    for race_id, group in base_df.groupby("race_id"):
        fav_idx = group["market_odds"].idxmin()
        fav_picks.add(fav_idx)
    mask_fav = base_df.index.isin(fav_picks)
    results.append(simulate_strategy("2. Always Favorite (lowest odds)", mask_fav, base_df))

    # ── 3a. Dominant Feature: Form Points (h_points_l5) ─────────────────────
    if "h_points_l5" in base_df.columns:
        form_picks = set()
        for race_id, group in base_df.groupby("race_id"):
            best_idx = group["h_points_l5"].idxmax()
            form_picks.add(best_idx)
        mask_form = base_df.index.isin(form_picks)
        results.append(simulate_strategy("3a. Best Form (h_points_l5)", mask_form, base_df))

    # ── 3b. Dominant Feature: Recent Speed (h_avg_speed_l5) ──────────────────
    if "h_avg_speed_l5" in base_df.columns:
        speed_picks = set()
        for race_id, group in base_df.groupby("race_id"):
            # Speed can be NaN — fill with 0 for ranking
            best_idx = group["h_avg_speed_l5"].fillna(0).idxmax()
            speed_picks.add(best_idx)
        mask_speed = base_df.index.isin(speed_picks)
        results.append(simulate_strategy("3b. Fastest (h_avg_speed_l5)", mask_speed, base_df))

    # ── 3c. Dominant Feature: Horse Speed Ratio (peak proximity) ─────────────
    if "h_speed_ratio" in base_df.columns:
        ratio_picks = set()
        for race_id, group in base_df.groupby("race_id"):
            best_idx = group["h_speed_ratio"].fillna(0).idxmax()
            ratio_picks.add(best_idx)
        mask_ratio = base_df.index.isin(ratio_picks)
        results.append(simulate_strategy("3c. Peak Form Ratio (h_speed_ratio)", mask_ratio, base_df))

    # ── 4. Model V3 Value Betting (best config: edge 15%, max odds 8) ─────────
    if all(c in sim_df.columns for c in ["market_odds", "fair_odds", "win"]):
        # Re-run with best config on sim_df (already 2025 only)
        mask_v3 = (
            (sim_df["market_odds"] > sim_df["fair_odds"] * 1.15)
            & (sim_df["market_odds"] <= 8.0)
        )
        # Simulate on sim_df (which has model probs)
        bets = mask_v3.sum()
        pnl = np.where(mask_v3, np.where(sim_df["win"] == 1, (sim_df["market_odds"] - 1) * STAKE, -STAKE), 0).sum()
        hits = sim_df[mask_v3]["win"].sum()
        roi = pnl / (bets * STAKE) * 100 if bets > 0 else 0
        results.append({
            "strategy": "4. V3 Model (edge 15%, MaxOdds ≤8)",
            "bets": int(bets),
            "staked_ft": int(bets * STAKE),
            "pnl_ft": int(pnl),
            "roi_pct": round(roi, 2),
            "hit_rate_pct": round(hits / bets * 100, 2) if bets > 0 else 0,
            "avg_odds": round(sim_df[mask_v3]["market_odds"].mean(), 2) if bets > 0 else 0,
        })

    # ── Print Results ─────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BASELINE STRATEGY COMPARISON — 2025 Test Set")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

    # Save for dashboard
    res_df.to_csv("data/baseline_results.csv", index=False)
    print("\nSaved to data/baseline_results.csv")
    return res_df


if __name__ == "__main__":
    run_baselines()
