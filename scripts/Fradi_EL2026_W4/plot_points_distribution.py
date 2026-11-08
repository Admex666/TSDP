"""
Ferencváros TC - UEFA Europa League 2026/27 Points Distribution Analysis
Visualizes the probability distribution of all possible point totals (0 to 12)
based on closing 1X2 betting odds across Matchweeks 1 to 4.
Adam Jakus styling: dark background (#3c3d3d), signature green (#5ECB43),
Nexa-ExtraLight typography, in full English.
"""

import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def create_points_distribution_plot(
    csv_path: str = "europa_league_2026_matches.csv",
    output_png: str = "fradi_points_distribution_adam_jakus.png"
):
    df = pd.read_csv(csv_path)

    # Filter Ferencvaros League Phase matches
    fin = df[
        (df["stage"] == "League Phase") &
        (df["status_type"] == "finished") &
        ((df["home_team_name"].str.contains("Ferenc")) | (df["away_team_name"].str.contains("Ferenc")))
    ].sort_values("round_nr")

    match_probs = []
    match_summaries = []

    for _, row in fin.iterrows():
        is_home = "Ferenc" in row["home_team_name"]
        opp = row["away_team_name"] if is_home else row["home_team_name"]
        o1, ox, o2 = row["odds_1"], row["odds_X"], row["odds_2"]
        
        # Proportional no-vig removal
        inv_sum = 1.0 / o1 + 1.0 / ox + 1.0 / o2
        p_w = (1.0 / o1 if is_home else 1.0 / o2) / inv_sum
        p_d = (1.0 / ox) / inv_sum
        p_l = 1.0 - p_w - p_d
        
        match_probs.append({"W": p_w, "D": p_d, "L": p_l})
        match_summaries.append({
            "round": int(row["round_nr"]),
            "opp": opp,
            "venue": "H" if is_home else "A",
            "odd": o1 if is_home else o2,
            "p_win": p_w * 100
        })

    # Calculate exact distribution for all 3^4 = 81 scenarios
    point_probs = {pts: 0.0 for pts in range(13)}
    
    for outcomes in product(["W", "D", "L"], repeat=4):
        prob = 1.0
        pts = 0
        for i, out in enumerate(outcomes):
            prob *= match_probs[i][out]
            if out == "W":
                pts += 3
            elif out == "D":
                pts += 1
        point_probs[pts] += prob

    dist_df = pd.DataFrame([
        {"points": pts, "prob": point_probs[pts], "pct": point_probs[pts] * 100}
        for pts in range(13)
    ])

    expected_pts = sum(dist_df["points"] * dist_df["prob"])
    prob_12_pts = dist_df.loc[dist_df["points"] == 12, "pct"].values[0]

    # Styling & Fonts
    background_color = "#3c3d3d"
    mycolor = "#5ECB43"  # Adam Jakus Green
    bar_default_color = "#4b5563"  # Slate grey
    bar_edge_default = "#6b7280"
    
    font_path_arvo = r"E:\Data\TSDP\helpers\Arvo-Regular.ttf"
    font_path_nexa = r"E:\Data\TSDP\helpers\Nexa-ExtraLight.ttf"
    
    arvo_prop = font_manager.FontProperties(fname=font_path_arvo) if os.path.exists(font_path_arvo) else None
    nexa_prop = font_manager.FontProperties(fname=font_path_nexa) if os.path.exists(font_path_nexa) else None

    # Setup Figure
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Colors for each bar (highlight 12 points)
    bar_colors = [mycolor if pts == 12 else bar_default_color for pts in dist_df["points"]]
    bar_edges = ["#ffffff" if pts == 12 else bar_edge_default for pts in dist_df["points"]]
    bar_widths = [0.75 if pts != 11 else 0.0 for pts in dist_df["points"]]

    bars = ax.bar(
        dist_df["points"], dist_df["pct"],
        color=bar_colors, edgecolor=bar_edges,
        width=0.72, linewidth=1.2, zorder=3
    )

    # Highlight 12 pts bar with glow / extra edge
    bars[12].set_linewidth(2.2)
    bars[12].set_edgecolor("#ffffff")

    # Bar value labels on top
    for pts, pct in zip(dist_df["points"], dist_df["pct"]):
        if pts == 11 or pct < 0.01:
            continue
        if pts == 12:
            ax.annotate(
                f"{pct:.2f}%\n(1 in 111)",
                (pts, pct),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontweight="bold",
                color=mycolor,
                fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=background_color, edgecolor=mycolor, lw=1.5, alpha=0.98),
                zorder=10
            )
        else:
            text_color = "white" if pct >= 4.0 else "#d1d5db"
            font_w = "bold" if pct >= 10.0 else "normal"
            ax.annotate(
                f"{pct:.1f}%",
                (pts, pct),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontweight=font_w,
                color=text_color,
                fontsize=10,
                zorder=5
            )

    # Vertical line for Expected Points (xPts)
    ax.axvline(expected_pts, color="#38bdf8", linestyle="--", linewidth=1.8, alpha=0.9, zorder=6)
    ax.annotate(
        f"Expected Points: {expected_pts:.2f} pts",
        xy=(expected_pts, 21.0),
        xytext=(expected_pts - 2.8, 22.2),
        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.5),
        color="#38bdf8",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=background_color, edgecolor="#38bdf8", lw=1.2, alpha=0.95),
        zorder=7
    )

    # Spines
    ax.spines["bottom"].set_color("white")
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("white")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid & Ticks
    ax.grid(True, axis="y", color="#555555", alpha=0.35, linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(axis="x", colors="white", labelsize=11.5, length=6, width=1.2)
    ax.tick_params(axis="y", colors="white", labelsize=11, length=6, width=1.2)

    ax.set_xticks(range(13))
    ax.set_xticklabels([f"{p} pts" if p != 11 else "" for p in range(13)], fontweight="bold")
    ax.set_xlim(-0.7, 13.0)
    ax.set_ylim(0, 25.0)

    ax.set_xlabel("Points after 4 Matchweeks", color="white", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Probability (%)", color="white", fontsize=12, fontweight="bold", labelpad=10)

    # Titles (in English)
    title_text = "FERENCVÁROSI TC: POINTS PROBABILITY DISTRIBUTION"
    subtitle_text = "Pre-match Odds-implied Distribution | UEFA Europa League 2026/2027 (Matchweeks 1-4)"
    
    if arvo_prop:
        fig.text(0.08, 0.955, title_text, color="white", fontsize=19, fontproperties=arvo_prop, ha="left")
    else:
        fig.text(0.08, 0.955, title_text, color="white", fontsize=19, fontweight="bold", ha="left")
        
    fig.text(0.08, 0.92, subtitle_text, color="#d1d5db", fontsize=11.5, ha="left")

    # Signature / Branding (ADAM JAKUS)
    if nexa_prop:
        fig.text(0.92, 0.025, "ADAM JAKUS", color=mycolor, fontsize=18, fontproperties=nexa_prop, ha="right", va="bottom")
    else:
        fig.text(0.92, 0.025, "ADAM JAKUS", color=mycolor, fontsize=18, fontweight="bold", ha="right", va="bottom")

    fig.text(
        0.08, 0.025,
        "Data: SofaScore",
        color="#9ca3af", fontsize=10.0, ha="left", va="bottom"
    )

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])
    plt.savefig(output_png, facecolor=background_color, edgecolor="none", dpi=300)
    plt.close()
    print(f"[Done] Points distribution plot created at '{output_png}'.")


if __name__ == "__main__":
    create_points_distribution_plot()
