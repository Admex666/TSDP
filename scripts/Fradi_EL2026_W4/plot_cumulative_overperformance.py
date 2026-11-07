"""
UEFA Europa League 2026/27 - Cumulative Expected Points (xPts) Overperformance
Custom Adam Jakus Styling: Dark background (#3c3d3d), signature green (#5ECB43),
Nexa-ExtraLight typography, in full English.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def create_adam_jakus_plot(
    csv_path: str = "europa_league_2026_matches.csv",
    output_png: str = "el_cumulative_overperformance_adam_jakus.png"
):
    df = pd.read_csv(csv_path)

    # Filter finished League Phase matches with odds
    fin = df[
        (df["stage"] == "League Phase") &
        (df["status_type"] == "finished") &
        (df["odds_1"].notna()) &
        (df["odds_X"].notna()) &
        (df["odds_2"].notna())
    ].copy()

    # 1. No-vig implied probabilities
    inv_1 = 1.0 / fin["odds_1"]
    inv_X = 1.0 / fin["odds_X"]
    inv_2 = 1.0 / fin["odds_2"]
    sum_inv = inv_1 + inv_X + inv_2

    fin["p_home"] = inv_1 / sum_inv
    fin["p_draw"] = inv_X / sum_inv
    fin["p_away"] = inv_2 / sum_inv

    # Expected points
    fin["xpts_home"] = 3.0 * fin["p_home"] + 1.0 * fin["p_draw"]
    fin["xpts_away"] = 3.0 * fin["p_away"] + 1.0 * fin["p_draw"]

    # Actual points
    def get_actual_pts(row):
        h, a = row["home_score"], row["away_score"]
        if h > a:
            return 3, 0
        elif h < a:
            return 0, 3
        else:
            return 1, 1

    pts = fin.apply(get_actual_pts, axis=1)
    fin["pts_home"] = [p[0] for p in pts]
    fin["pts_away"] = [p[1] for p in pts]

    # Team round-by-round records
    team_records = []
    for _, row in fin.iterrows():
        rnd = int(row["round_nr"])
        team_records.append({
            "team": row["home_team_name"],
            "round": rnd,
            "pts": row["pts_home"],
            "xpts": row["xpts_home"],
            "diff": row["pts_home"] - row["xpts_home"]
        })
        team_records.append({
            "team": row["away_team_name"],
            "round": rnd,
            "pts": row["pts_away"],
            "xpts": row["xpts_away"],
            "diff": row["pts_away"] - row["xpts_away"]
        })

    df_tr = pd.DataFrame(team_records).sort_values(by=["team", "round"])
    df_tr["cum_pts"] = df_tr.groupby("team")["pts"].cumsum()
    df_tr["cum_xpts"] = df_tr.groupby("team")["xpts"].cumsum()
    df_tr["cum_diff"] = df_tr["cum_pts"] - df_tr["cum_xpts"]

    rounds = sorted(df_tr["round"].unique())
    last_round = max(rounds)

    # Final round rankings
    final_rank = df_tr[df_tr["round"] == last_round].sort_values(by="cum_diff", ascending=False)
    
    # Fonts & Colors
    background_color = "#3c3d3d"
    mycolor = "#5ECB43"  # Adam Jakus Green
    
    font_path_arvo = r"E:\Data\TSDP\helpers\Arvo-Regular.ttf"
    font_path_nexa = r"E:\Data\TSDP\helpers\Nexa-ExtraLight.ttf"
    
    arvo_prop = font_manager.FontProperties(fname=font_path_arvo) if os.path.exists(font_path_arvo) else None
    nexa_prop = font_manager.FontProperties(fname=font_path_nexa) if os.path.exists(font_path_nexa) else None

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Zero expectation baseline
    ax.axhline(0, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.55, zorder=2)
    ax.text(
        0.85, 0.15, "Odds Expectation (0 pts)",
        color="#cccccc", fontsize=9.5, fontstyle="italic", alpha=0.8, va="bottom"
    )

    target_team = "Ferencváros TC"
    top_labels = ["Bournemouth", "Royale Union Saint-Gilloise", "Benfica"]
    bottom_labels = ["AC Milan", "Lillestrøm SK", "TSG Hoffenheim"]

    palette_top = {
        "Bournemouth": "#38bdf8",              # Sky blue
        "Royale Union Saint-Gilloise": "#fbbf24",# Warm amber
        "Benfica": "#f87171"                   # Soft coral red
    }
    palette_bottom = {
        "AC Milan": "#ef4444",                 # Crimson red
        "Lillestrøm SK": "#f97316",            # Orange red
        "TSG Hoffenheim": "#fb923c"            # Soft orange
    }

    # 1. Background grey lines for all other teams
    for team, grp in df_tr.groupby("team"):
        if team == target_team or team in palette_top or team in palette_bottom:
            continue
        grp = grp.sort_values("round")
        ax.plot(grp["round"], grp["cum_diff"], color="#888888", linewidth=1.0, alpha=0.28, zorder=3)

    # 2. Highlight Top Teams
    for team, col in palette_top.items():
        grp = df_tr[df_tr["team"] == team].sort_values("round")
        ax.plot(grp["round"], grp["cum_diff"], color=col, linewidth=2.0, alpha=0.9, marker="o", markersize=4.5, zorder=5)
        last_val = grp[grp["round"] == last_round]["cum_diff"].values[0]
        short_name = "Union SG" if "Royale" in team else team
        ax.text(
            last_round + 0.06, last_val, f"{short_name} (+{last_val:.2f} pts)",
            color=col, fontsize=9.5, fontweight="bold", va="center", zorder=6
        )

    # 3. Highlight Bottom Teams
    for team, col in palette_bottom.items():
        grp = df_tr[df_tr["team"] == team].sort_values("round")
        ax.plot(grp["round"], grp["cum_diff"], color=col, linewidth=2.0, alpha=0.9, marker="o", markersize=4.5, zorder=5)
        last_val = grp[grp["round"] == last_round]["cum_diff"].values[0]
        ax.text(
            last_round + 0.06, last_val, f"{team} ({last_val:.2f} pts)",
            color=col, fontsize=9.5, fontweight="bold", va="center", zorder=6
        )

    # 4. Highlight Ferencváros TC (Front and Center)
    fradi_grp = df_tr[df_tr["team"] == target_team].sort_values("round")
    ax.plot(
        fradi_grp["round"], fradi_grp["cum_diff"],
        color=mycolor, linewidth=4.0, marker="s", markersize=9,
        markeredgecolor="#ffffff", markeredgewidth=1.2,
        label="Ferencváros TC (#1 Overperformer)", zorder=10
    )

    # Callout values for Ferencváros
    for _, row in fradi_grp.iterrows():
        val = row["cum_diff"]
        r = row["round"]
        offset_y = 12 if r != 3 else 14
        ax.annotate(
            f"+{val:.2f} pts",
            (r, val),
            textcoords="offset points",
            xytext=(0, offset_y),
            ha="center",
            fontweight="bold",
            color=mycolor,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=background_color, edgecolor=mycolor, lw=1.2, alpha=0.95),
            zorder=11
        )

    # End label for Ferencváros
    last_fradi_val = fradi_grp[fradi_grp["round"] == last_round]["cum_diff"].values[0]
    ax.text(
        last_round + 0.06, last_fradi_val, f"Ferencváros TC (+{last_fradi_val:.2f} pts)",
        color=mycolor, fontsize=12, fontweight="bold", va="center", zorder=12
    )

    # Spine styling
    ax.spines["bottom"].set_color("white")
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("white")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid & Ticks
    ax.grid(True, color="#555555", alpha=0.35, linestyle="--", linewidth=0.8, zorder=1)
    ax.tick_params(axis="x", colors="white", labelsize=12, length=6, width=1.2)
    ax.tick_params(axis="y", colors="white", labelsize=11, length=6, width=1.2)

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Matchweek {r}" for r in rounds], fontweight="bold")
    ax.set_xlim(min(rounds) - 0.25, last_round + 1.25)
    ax.set_ylim(-4.5, 7.8)

    ax.set_ylabel("Cumulative Overperformance (Actual Pts − Expected Pts)", color="white", fontsize=12, fontweight="bold", labelpad=12)

    # Titles (in English)
    title_text = "UEFA EUROPA LEAGUE 2026/27"
    subtitle_text = "Cumulative Points Overperformance vs Odds Expectation (xPts) | Matchweeks 1–4"
    
    if arvo_prop:
        fig.text(0.12, 0.955, title_text, color="white", fontsize=20, fontproperties=arvo_prop, ha="left")
    else:
        fig.text(0.12, 0.955, title_text, color="white", fontsize=20, fontweight="bold", ha="left")
        
    fig.text(0.12, 0.92, subtitle_text, color="#d1d5db", fontsize=12, ha="left")

    # Legend
    legend = ax.legend(
        loc="upper left",
        facecolor=background_color,
        edgecolor=mycolor,
        fontsize=10.5,
        labelcolor="white",
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(1.2)

    # Signature / Branding (ADAM JAKUS)
    if nexa_prop:
        fig.text(0.90, 0.02, "ADAM JAKUS", color=mycolor, fontsize=18, fontproperties=nexa_prop, ha="right", va="bottom")
    else:
        fig.text(0.90, 0.02, "ADAM JAKUS", color=mycolor, fontsize=18, fontweight="bold", ha="right", va="bottom")

    fig.text(
        0.12, 0.02,
        "Data: SofaScore API | Expected points computed from 1X2 odds with proportional margin removal (no-vig)",
        color="#9ca3af", fontsize=9.5, ha="left", va="bottom"
    )

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])
    plt.savefig(output_png, facecolor=background_color, edgecolor="none", dpi=300)
    plt.close()
    print(f"[Done] Plot generated successfully at '{output_png}'.")


if __name__ == "__main__":
    create_adam_jakus_plot()
