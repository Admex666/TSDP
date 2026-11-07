"""
Europa League 2026/27 - Expected Points (xPts) and Overperformance Analysis
Computes no-vig implied probabilities, expected points (xPts), and compares
with actual points. Plots rolling average overperformance by round.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_expected_points(csv_path: str = "europa_league_2026_matches.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Filter for finished League Phase matches with valid odds
    fin = df[
        (df["stage"] == "League Phase") &
        (df["status_type"] == "finished") &
        (df["odds_1"].notna()) &
        (df["odds_X"].notna()) &
        (df["odds_2"].notna())
    ].copy()
    
    if fin.empty:
        raise ValueError("No finished League Phase matches with odds found!")

    # 1. No-vig implied probabilities (proportional margin removal)
    inv_1 = 1.0 / fin["odds_1"]
    inv_X = 1.0 / fin["odds_X"]
    inv_2 = 1.0 / fin["odds_2"]
    sum_inv = inv_1 + inv_X + inv_2
    
    fin["p_home"] = inv_1 / sum_inv
    fin["p_draw"] = inv_X / sum_inv
    fin["p_away"] = inv_2 / sum_inv

    # 2. Expected points per match
    fin["xpts_home"] = 3.0 * fin["p_home"] + 1.0 * fin["p_draw"]
    fin["xpts_away"] = 3.0 * fin["p_away"] + 1.0 * fin["p_draw"]

    # 3. Actual points per match
    def get_actual_pts(row):
        h = row["home_score"]
        a = row["away_score"]
        if h > a:
            return 3, 0
        elif h < a:
            return 0, 3
        else:
            return 1, 1

    pts = fin.apply(get_actual_pts, axis=1)
    fin["pts_home"] = [p[0] for p in pts]
    fin["pts_away"] = [p[1] for p in pts]

    # 4. Long format table by team & round
    team_records = []
    for _, row in fin.iterrows():
        rnd = int(row["round_nr"])
        team_records.append({
            "team": row["home_team_name"],
            "round": rnd,
            "pts": row["pts_home"],
            "xpts": row["xpts_home"],
            "diff": row["pts_home"] - row["xpts_home"],
            "is_home": True,
            "opponent": row["away_team_name"]
        })
        team_records.append({
            "team": row["away_team_name"],
            "round": rnd,
            "pts": row["pts_away"],
            "xpts": row["xpts_away"],
            "diff": row["pts_away"] - row["xpts_away"],
            "is_home": False,
            "opponent": row["home_team_name"]
        })

    df_tr = pd.DataFrame(team_records).sort_values(by=["team", "round"])
    
    # Cumulative & rolling statistics
    df_tr["cum_pts"] = df_tr.groupby("team")["pts"].cumsum()
    df_tr["cum_xpts"] = df_tr.groupby("team")["xpts"].cumsum()
    df_tr["cum_diff"] = df_tr["cum_pts"] - df_tr["cum_xpts"]
    df_tr["rolling_avg_diff"] = df_tr.groupby("team")["diff"].expanding().mean().reset_index(level=0, drop=True)

    # 5. Summary Table
    summary = df_tr.groupby("team").agg(
        played=("round", "count"),
        actual_pts=("pts", "sum"),
        expected_pts=("xpts", "sum"),
        total_overperformance=("diff", "sum"),
        avg_overperformance_per_match=("diff", "mean")
    ).sort_values(by="total_overperformance", ascending=False).reset_index()

    summary["actual_pts"] = summary["actual_pts"].astype(int)
    summary["expected_pts"] = summary["expected_pts"].round(2)
    summary["total_overperformance"] = summary["total_overperformance"].round(2)
    summary["avg_overperformance_per_match"] = summary["avg_overperformance_per_match"].round(2)

    return fin, df_tr, summary


def plot_rolling_overperformance(df_tr: pd.DataFrame, output_image: str = "el_expected_points_overperformance.png"):
    """
    Generate high quality time series plots of rolling average and cumulative overperformance.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    
    rounds = sorted(df_tr["round"].unique())
    last_round = max(rounds)
    
    # Identify top 3 overperformers, bottom 3 underperformers, and Ferencváros
    last_stats = df_tr[df_tr["round"] == last_round].sort_values(by="cum_diff", ascending=False)
    
    top_teams = last_stats.head(4)["team"].tolist()
    bottom_teams = last_stats.tail(3)["team"].tolist()
    target_team = "Ferencváros TC"
    
    # Colors
    color_fradi = "#008751"  # Fradi Green
    color_top = "#1f77b4"    # Blue
    color_bottom = "#d62728" # Red
    color_bg = "#b0bec5"     # Muted grey
    
    # 1. Left Panel: Rolling Average Overperformance (pts / match)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Odds elvárás (0)")
    
    for team, group in df_tr.groupby("team"):
        group = group.sort_values("round")
        if team == target_team:
            continue
        elif team in top_teams:
            ax1.plot(group["round"], group["rolling_avg_diff"], color=color_top, linewidth=1.8, alpha=0.85, marker="o", markersize=4)
            # Label at end
            last_val = group[group["round"] == last_round]["rolling_avg_diff"].values[0]
            ax1.text(last_round + 0.05, last_val, f"{team} (+{last_val:.2f})", fontsize=8.5, color=color_top, va="center")
        elif team in bottom_teams:
            ax1.plot(group["round"], group["rolling_avg_diff"], color=color_bottom, linewidth=1.8, alpha=0.85, marker="o", markersize=4)
            last_val = group[group["round"] == last_round]["rolling_avg_diff"].values[0]
            ax1.text(last_round + 0.05, last_val, f"{team} ({last_val:.2f})", fontsize=8.5, color=color_bottom, va="center")
        else:
            ax1.plot(group["round"], group["rolling_avg_diff"], color=color_bg, linewidth=0.9, alpha=0.35)

    # Highlight Ferencváros on Left Panel
    fradi_grp = df_tr[df_tr["team"] == target_team].sort_values("round")
    ax1.plot(fradi_grp["round"], fradi_grp["rolling_avg_diff"], color=color_fradi, linewidth=3.5, marker="s", markersize=8, label="Ferencváros TC", zorder=10)
    for _, row in fradi_grp.iterrows():
        ax1.annotate(
            f"+{row['rolling_avg_diff']:.2f}",
            (row["round"], row["rolling_avg_diff"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color=color_fradi,
            fontsize=9
        )
    
    ax1.set_title("Gördülő átlagos felülteljesítés (pont / mérkőzés)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Forduló", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Átlagos felülteljesítés (Valós pont - Várható pont)", fontsize=11)
    ax1.set_xticks(rounds)
    ax1.set_xticklabels([f"{r}. forduló" for r in rounds])
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_xlim(min(rounds) - 0.2, last_round + 1.2)
    
    # 2. Right Panel: Cumulative Total Overperformance (pts)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    
    for team, group in df_tr.groupby("team"):
        group = group.sort_values("round")
        if team == target_team:
            continue
        elif team in top_teams:
            ax2.plot(group["round"], group["cum_diff"], color=color_top, linewidth=1.8, alpha=0.85, marker="o", markersize=4)
            last_val = group[group["round"] == last_round]["cum_diff"].values[0]
            ax2.text(last_round + 0.05, last_val, f"{team} (+{last_val:.2f}p)", fontsize=8.5, color=color_top, va="center")
        elif team in bottom_teams:
            ax2.plot(group["round"], group["cum_diff"], color=color_bottom, linewidth=1.8, alpha=0.85, marker="o", markersize=4)
            last_val = group[group["round"] == last_round]["cum_diff"].values[0]
            ax2.text(last_round + 0.05, last_val, f"{team} ({last_val:.2f}p)", fontsize=8.5, color=color_bottom, va="center")
        else:
            ax2.plot(group["round"], group["cum_diff"], color=color_bg, linewidth=0.9, alpha=0.35)

    # Highlight Ferencváros on Right Panel
    ax2.plot(fradi_grp["round"], fradi_grp["cum_diff"], color=color_fradi, linewidth=3.5, marker="s", markersize=8, label="Ferencváros TC", zorder=10)
    for _, row in fradi_grp.iterrows():
        ax2.annotate(
            f"+{row['cum_diff']:.2f}p",
            (row["round"], row["cum_diff"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color=color_fradi,
            fontsize=9
        )
    
    ax2.set_title("Összesített kumulált felülteljesítés (összes pont)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Forduló", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Összes felülteljesítés (Valós pont - Várható pont)", fontsize=11)
    ax2.set_xticks(rounds)
    ax2.set_xticklabels([f"{r}. forduló" for r in rounds])
    ax2.set_xlim(min(rounds) - 0.2, last_round + 1.2)
    
    plt.suptitle("UEFA Európa Liga 2026/27 (Alapszakasz 1–4. forduló)\nOdds-alapú Várható Pont (xPts) Felülteljesítés", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] Successfully created '{output_image}'.")


def main():
    csv_file = "europa_league_2026_matches.csv"
    summary_file = "el_expected_points_summary.csv"
    plot_file = "el_expected_points_overperformance.png"
    
    fin, df_tr, summary = calculate_expected_points(csv_file)
    
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"[Summary saved] Table exported to '{summary_file}'.")
    
    plot_rolling_overperformance(df_tr, plot_file)
    
    print("\n=== TOP 10 FELÜLTELJESÍTŐ CSAPAT (xPts vs Valós pontok) ===")
    print(summary.head(10).to_string(index=False))
    
    print("\n=== TOP 5 ALULTELJESÍTŐ CSAPAT ===")
    print(summary.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
