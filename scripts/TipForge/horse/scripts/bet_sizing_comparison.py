"""
Bet-Sizing Comparison for Model V3
====================================
Strategies tested on 2025 test set with different bet-sizing methods:

  A. Flat Stake         — fixed 1000 Ft per bet
  B. Full Kelly         — f* = (b*p - q) / b  (aggressive)
  C. Half Kelly         — 0.5 * f*            (moderate)
  D. Quarter Kelly      — 0.25 * f*           (conservative)
  E. Proportional Edge  — stake proportional to edge %, capped

All tested under:
  - Edge threshold grid: 5%, 10%, 15%, 20%
  - Max market odds cap: 8.0 (proven best from diagnostic)
  - Starting bankroll: 100,000 Ft
"""

import pandas as pd
import numpy as np
import os

BANKROLL     = 100_000   # Ft starting bankroll
MAX_BET_FRAC = 0.20      # Never risk more than 20% of bankroll on one bet
MIN_BET      = 200       # Minimum bet size (Ft)
FIXED_STAKE  = 1_000     # Flat stake fallback (Ft)
MAX_ODDS     = 8.0       # Max market odds (from diagnostic)


def kelly_fraction(prob, odds):
    """Full Kelly fraction. prob=win prob, odds=decimal market odds."""
    b = odds - 1.0  # net profit per unit
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(f, 0.0)  # never negative


def simulate_bet_sizing(df, name, sizing_fn, edge_min, bankroll=BANKROLL):
    """
    sizing_fn(row) -> fraction of current bankroll to stake.
    Returns a summary dict.
    """
    df = df.dropna(subset=["market_odds", "fair_odds", "prob_norm", "win"]).copy().reset_index(drop=True)

    bank = bankroll
    pnl_series = []
    bank_series = []
    bets = 0
    wins = 0
    total_staked = 0

    for _, row in df.iterrows():
        edge = (row["market_odds"] / row["fair_odds"]) - 1
        if edge < edge_min or row["market_odds"] > MAX_ODDS:
            continue

        frac = sizing_fn(row)
        if frac <= 0:
            continue

        # Cap per-bet fraction
        frac = min(frac, MAX_BET_FRAC)
        stake = max(round(bank * frac, 0), MIN_BET)
        stake = min(stake, bank)  # can't bet more than bankroll

        if row["win"] == 1:
            profit = (row["market_odds"] - 1) * stake
            bank += profit
            wins += 1
        else:
            bank -= stake
            profit = -stake

        pnl_series.append({"stake": stake, "profit": profit, "bank": bank})
        total_staked += stake
        bets += 1

        if bank <= 0:
            break  # bankrupt

    if bets == 0 or total_staked == 0:
        return {
            "name": name, "edge_min_pct": edge_min * 100,
            "bets": 0, "final_bank": bankroll, "total_pnl": 0,
            "roi_on_staked": 0, "bank_growth": 0, "hit_rate": 0, "bankrupt": False
        }

    final_bank = bank
    total_pnl  = final_bank - bankroll
    roi_staked = total_pnl / total_staked * 100
    growth     = (final_bank / bankroll - 1) * 100

    return {
        "name": name,
        "edge_min_pct": edge_min * 100,
        "bets": bets,
        "wins": wins,
        "hit_rate_pct": round(wins / bets * 100, 2),
        "total_staked": int(total_staked),
        "total_pnl": int(total_pnl),
        "roi_on_staked_pct": round(roi_staked, 2),
        "bank_growth_pct": round(growth, 2),
        "final_bank": int(final_bank),
        "bankrupt": bank <= 0,
        "pnl_series": pnl_series,
    }


def run_bet_sizing():
    sim_path  = "data/simulation_results.csv"
    if not os.path.exists(sim_path):
        print("simulation_results.csv not found. Run simulate_value_betting.py first.")
        return

    print("Loading simulation data...")
    df = pd.read_csv(sim_path)
    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    all_results = []

    for edge_min in [0.05, 0.10, 0.15, 0.20]:

        # ── A. Flat Stake ─────────────────────────────────────────────────────
        def flat_fn(row, _e=edge_min):
            return FIXED_STAKE / BANKROLL  # constant fraction

        r = simulate_bet_sizing(df, "A. Flat (1000 Ft)", flat_fn, edge_min)
        all_results.append(r)

        # ── B. Full Kelly ─────────────────────────────────────────────────────
        def full_kelly(row):
            return kelly_fraction(row["prob_norm"], row["market_odds"])

        r = simulate_bet_sizing(df, "B. Full Kelly", full_kelly, edge_min)
        all_results.append(r)

        # ── C. Half Kelly ─────────────────────────────────────────────────────
        def half_kelly(row):
            return 0.5 * kelly_fraction(row["prob_norm"], row["market_odds"])

        r = simulate_bet_sizing(df, "C. Half Kelly", half_kelly, edge_min)
        all_results.append(r)

        # ── D. Quarter Kelly ──────────────────────────────────────────────────
        def quarter_kelly(row):
            return 0.25 * kelly_fraction(row["prob_norm"], row["market_odds"])

        r = simulate_bet_sizing(df, "D. Quarter Kelly", quarter_kelly, edge_min)
        all_results.append(r)

        # ── E. Proportional Edge (edge% * 0.5 of bankroll) ───────────────────
        def prop_edge(row):
            edge = max((row["market_odds"] / row["fair_odds"]) - 1, 0)
            return edge * 0.5  # bet up to 50% of edge fraction

        r = simulate_bet_sizing(df, "E. Prop. Edge (50%)", prop_edge, edge_min)
        all_results.append(r)

    # ── Print Summary ─────────────────────────────────────────────────────────
    summary_cols = ["name", "edge_min_pct", "bets", "hit_rate_pct",
                    "total_pnl", "roi_on_staked_pct", "bank_growth_pct", "final_bank", "bankrupt"]
    rows = [{k: r.get(k, None) for k in summary_cols} for r in all_results]
    res_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"BET-SIZING COMPARISON | Bankroll: {BANKROLL:,} Ft | MaxOdds: {MAX_ODDS}")
    print("=" * 100)
    print(res_df.to_string(index=False))
    print("=" * 100)

    # Save summary + per-bet series for the best config
    res_df.to_csv("data/bet_sizing_results.csv", index=False)
    print("\nSaved summary → data/bet_sizing_results.csv")

    # Find best by bank_growth and save its pnl_series
    best = max(all_results, key=lambda r: r.get("bank_growth_pct", -999))
    print(f"\nBest config: {best['name']} @ Edge ≥{best['edge_min_pct']:.0f}% → "
          f"Bank growth: {best['bank_growth_pct']:+.2f}%  |  "
          f"Final bank: {best['final_bank']:,} Ft")

    # Save each strategy's bankroll curve for dashboard
    curves = {}
    for r in all_results:
        key = f"{r['name']} | Edge {r['edge_min_pct']:.0f}%"
        if r.get("pnl_series"):
            curves[key] = [p["bank"] for p in r["pnl_series"]]

    # Save as long-format CSV for easy charting
    curve_rows = []
    for label, banks in curves.items():
        for i, b in enumerate(banks):
            curve_rows.append({"strategy": label, "bet_num": i + 1, "bankroll": b})
    if curve_rows:
        curve_df = pd.DataFrame(curve_rows)
        curve_df.to_csv("data/bet_sizing_curves.csv", index=False)
        print("Saved bankroll curves → data/bet_sizing_curves.csv")

    return res_df


if __name__ == "__main__":
    run_bet_sizing()
