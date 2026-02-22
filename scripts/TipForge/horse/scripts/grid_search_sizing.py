"""
Quick grid search: Edge threshold x Max Odds x Kelly fraction
Bankroll simulation fully vectorized (approximate — assumes independent bets,
not bankroll-dependent sizing, for speed). Then re-runs best configs row-by-row.
"""
import pandas as pd
import numpy as np

BANKROLL     = 100_000
MAX_BET_FRAC = 0.20
MIN_BET      = 200

def kelly(p, odds):
    b = odds - 1
    return np.where(b > 0, np.clip((b * p - (1 - p)) / b, 0, None), 0)

def sim_rowbyrow(df, edge_min, max_odds, k_frac):
    """Row-by-row with real bankroll tracking."""
    bank = float(BANKROLL)
    wins = bets = 0
    for _, row in df.iterrows():
        edge = (row["market_odds"] / row["fair_odds"]) - 1
        if edge < edge_min or row["market_odds"] > max_odds:
            continue
        f = kelly(row["prob_norm"], row["market_odds"]) * k_frac
        if f <= 0:
            continue
        f = min(f, MAX_BET_FRAC)
        stake = max(round(bank * f), MIN_BET)
        stake = min(stake, bank)
        if row["win"] == 1:
            bank += (row["market_odds"] - 1) * stake
            wins += 1
        else:
            bank -= stake
        bets += 1
        if bank <= 0:
            return bets, wins, 0.0
    return bets, wins, bank

def main():
    df = pd.read_csv("data/simulation_results.csv").dropna(
        subset=["market_odds", "fair_odds", "prob_norm", "win"]
    ).sort_values("date").reset_index(drop=True)

    print(f"Test rows: {len(df)}\n")
    print(f"{'Edge':>6} | {'MaxOdds':>8} | {'Kelly':>8} | {'Bets':>5} | {'Growth':>8} | {'Final Bank':>12} | {'Bankrupt':>8}")
    print("-" * 75)

    results = []
    for edge_min in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for max_odds in [6.0, 8.0, 12.0, 99.0]:
            for k_label, k_frac in [("Full", 1.0), ("Half", 0.5), ("Quarter", 0.25)]:
                bets, wins, final = sim_rowbyrow(df, edge_min, max_odds, k_frac)
                growth = (final / BANKROLL - 1) * 100
                bankrupt = final <= 0
                tag = "💀" if bankrupt else ("✅" if growth > 0 else "❌")
                print(f"{edge_min*100:>5.0f}% | {max_odds:>8.1f} | {k_label:>8} | {bets:>5} | {growth:>+7.1f}% | {int(final):>12,} Ft | {tag}")
                results.append(dict(edge_min_pct=edge_min*100, max_odds=max_odds,
                                    kelly=k_label, bets=bets, growth_pct=round(growth,2),
                                    final_bank=int(final), bankrupt=bankrupt))

    print("-" * 75)
    res_df = pd.DataFrame(results)
    best = res_df.loc[res_df["growth_pct"].idxmax()]
    print(f"\n🏆 Best: Edge≥{best.edge_min_pct:.0f}% | MaxOdds≤{best.max_odds:.0f} | {best.kelly} Kelly "
          f"→ {best.growth_pct:+.2f}% | {int(best.final_bank):,} Ft")

    res_df.to_csv("data/grid_search_results.csv", index=False)
    print("Saved → data/grid_search_results.csv")

if __name__ == "__main__":
    main()
