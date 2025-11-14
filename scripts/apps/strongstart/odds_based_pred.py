#%% Imports
import os
import sys
import pandas as pd
import glob

#%% Working directory beállítása
allowed_dirs = [
    r"C:\Users\Adam\..Data",
    r"C:\Users\Adam\.Data files"
]
found = False
for path in allowed_dirs:
    if os.path.isdir(path):
        os.chdir(path)
        print(f"✅ Working directory set to: {path}")
        found = True
        break

if not found:
    print("❌ No valid working directory found.")
    sys.exit(1)

output_dir = "TSDP/datafiles"
os.makedirs(output_dir, exist_ok=True)

#%% Functions
# Odds → probability normalizálás
def odds_to_probs(home_odds, draw_odds, away_odds):
    """Convert 1X2 odds to normalized probabilities (removing overround)."""
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return None, None, None
    
    imp_home = 1 / home_odds
    imp_draw = 1 / draw_odds
    imp_away = 1 / away_odds
    total = imp_home + imp_draw + imp_away
    
    # Normalization
    p_home = imp_home / total
    p_draw = imp_draw / total
    p_away = imp_away / total
    
    return p_home, p_draw, p_away

# Expected points calculation
def expected_points(p_home, p_draw, p_away, is_home=True):
    """Calculate expected points from normalized probabilities."""
    if is_home:
        return p_home * 3 + p_draw * 1 + p_away * 0
    else:
        return p_away * 3 + p_draw * 1 + p_home * 0

#%% Load and process all CSV files
all_files = glob.glob(os.path.join(output_dir, "*_E0.csv"))
season_results = []

for file in all_files:
    season_code = os.path.basename(file).split("_")[0]
    df = pd.read_csv(file)
    
    # Only keep necessary columns
    cols_needed = ["Date", "HomeTeam", "AwayTeam", "FTR", "AvgH", "AvgD", "AvgA"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns in {file}: {missing}")
        continue
    
    # Calculate normalized probabilities
    probs = df.apply(lambda row: odds_to_probs(row["AvgH"], row["AvgD"], row["AvgA"]), axis=1)
    df[["p_home", "p_draw", "p_away"]] = pd.DataFrame(probs.tolist(), index=df.index)
    
    # Expected points for each match
    df["exp_home_pts"] = df.apply(lambda r: expected_points(r["p_home"], r["p_draw"], r["p_away"], is_home=True), axis=1)
    df["exp_away_pts"] = df.apply(lambda r: expected_points(r["p_home"], r["p_draw"], r["p_away"], is_home=False), axis=1)
    
    # Actual points
    df["act_home_pts"] = df["FTR"].map({"H": 3, "D": 1, "A": 0})
    df["act_away_pts"] = df["FTR"].map({"A": 3, "D": 1, "H": 0})
    
    # Aggregate by team
    home_stats = df.groupby("HomeTeam").agg(
        exp_pts=("exp_home_pts", "sum"),
        act_pts=("act_home_pts", "sum")
    )
    away_stats = df.groupby("AwayTeam").agg(
        exp_pts=("exp_away_pts", "sum"),
        act_pts=("act_away_pts", "sum")
    )
    
    season_table = home_stats.add(away_stats, fill_value=0)
    season_table["diff"] = season_table["act_pts"] - season_table["exp_pts"]
    season_table["Season"] = season_code
    season_table = season_table.reset_index().rename(columns={"index": "Team"})
    
    season_results.append(season_table)

#%% Combine all seasons
final_df = pd.concat(season_results, ignore_index=True)
final_df.rename(columns={'HomeTeam': 'Team'}, inplace=True)

# Sort and display
final_df = final_df[["Season", "Team", "exp_pts", "act_pts", "diff"]]
final_df = final_df.sort_values(["Season", "act_pts"], ascending=[True, False])

print(final_df.head(20))

#%% Extra metrics per season

from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

# 1. RMSE a pontszámokra (globális és szezonos bontásban)
rmse_global = np.sqrt(mean_squared_error(final_df["act_pts"], final_df["exp_pts"]))
print(f"📊 RMSE (points, overall): {rmse_global:.3f}")

print("📊 RMSE by season:")
for season, group in final_df.groupby("Season"):
    rmse_season = np.sqrt(mean_squared_error(group["act_pts"], group["exp_pts"]))
    print(f"  {season}: {rmse_season:.3f}")

# 2. Spearman rank correlation (helyezés sorrendek)
spearman_results = []
for season, group in final_df.groupby("Season"):
    group = group.copy()
    group["rank_act"] = group["act_pts"].rank(ascending=False, method="min")
    group["rank_exp"] = group["exp_pts"].rank(ascending=False, method="min")
    corr, _ = spearmanr(group["rank_act"], group["rank_exp"])
    spearman_results.append((season, corr))

# Globális Spearman (összes szezon egyben – nem annyira értelmes, de kiszámítjuk)
global_corr, _ = spearmanr(
    final_df["act_pts"].rank(ascending=False, method="min"),
    final_df["exp_pts"].rank(ascending=False, method="min")
)
print(f"🔗 Spearman rank correlation (overall): {global_corr:.3f}")

print("🔗 Spearman rank correlation by season:")
for season, corr in spearman_results:
    print(f"  {season}: {corr:.3f}")

# 3. Brier score a mérkőzéskimenetelekre (globális és szezonos)
def brier_score_row(row):
    outcome = row["FTR"]
    probs = np.array([row["p_home"], row["p_draw"], row["p_away"]])
    if outcome == "H":
        actual = np.array([1, 0, 0])
    elif outcome == "D":
        actual = np.array([0, 1, 0])
    elif outcome == "A":
        actual = np.array([0, 0, 1])
    else:
        return np.nan
    return np.mean((probs - actual) ** 2)

brier_scores_season = {}
all_briers = []

for file in all_files:
    season_code = os.path.basename(file).split("_")[0]
    df = pd.read_csv(file)
    if not {"AvgH", "AvgD", "AvgA", "FTR"}.issubset(df.columns):
        continue
    probs = df.apply(lambda row: odds_to_probs(row["AvgH"], row["AvgD"], row["AvgA"]), axis=1)
    df[["p_home", "p_draw", "p_away"]] = pd.DataFrame(probs.tolist(), index=df.index)
    df["brier"] = df.apply(brier_score_row, axis=1)
    season_brier = df["brier"].mean()
    brier_scores_season[season_code] = season_brier
    all_briers.extend(df["brier"].dropna().tolist())

# Globális Brier score
global_brier = np.mean(all_briers)
print(f"🎯 Brier score (overall): {global_brier:.4f}")

print("🎯 Brier score by season:")
for season_code, score in brier_scores_season.items():
    print(f"  {season_code}: {score:.4f}")

#%% Visualization: Expected vs Actual Points (Over/Under Performance)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib import font_manager
import os

# Convert season codes to nicer labels
def season_label(code):
    if len(code) == 4:  # e.g. "1920"
        start_year = 2000 + int(code[:2]) if int(code[:2]) < 50 else 1900 + int(code[:2])
        end_year = 2000 + int(code[2:]) if int(code[2:]) < 50 else 1900 + int(code[2:])
        return f"{start_year}-{str(end_year)[-2:]}"
    return code

final_df["SeasonLabel"] = final_df["Season"].apply(season_label)

# --- Styling to match your charts ---
background_color = '#3c3d3d'
mycolor = '#5ECB43'
my_font_path = os.getcwd() + r'\TSDP\Athletic\Nexa-ExtraLight.ttf'
my_font_props = font_manager.FontProperties(fname=my_font_path)

plt.figure(figsize=(10, 8), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)

# Seasons and colors (greens)
seasons = sorted(final_df["SeasonLabel"].unique())
colors = cm.Greens(np.linspace(0.4, 0.9, len(seasons)))

for i, season in enumerate(seasons):
    df_season = final_df[final_df["SeasonLabel"] == season]
    plt.scatter(
        df_season["exp_pts"], 
        df_season["act_pts"], 
        label=season, 
        alpha=0.9,
        s=80,
        color=colors[i],
        edgecolor='white',
        linewidth=0.5
    )

    # Label top over- and under-performers
    top_over = df_season.loc[df_season["diff"].idxmax()]
    top_under = df_season.loc[df_season["diff"].idxmin()]
    
    over_label = f"{top_over['Team']} ({season})"
    under_label = f"{top_under['Team']} ({season})"
    
    plt.text(top_over["exp_pts"]+0.3, top_over["act_pts"], over_label, fontsize=8, color="white")
    plt.text(top_under["exp_pts"]+0.3, top_under["act_pts"], under_label, fontsize=8, color="white")

# Perfect prediction diagonal
min_pts = min(final_df["exp_pts"].min(), final_df["act_pts"].min()) - 1
max_pts = max(final_df["exp_pts"].max(), final_df["act_pts"].max()) + 1
plt.plot([min_pts, max_pts], [min_pts, max_pts], linestyle='--', color='white', linewidth=1)

# Feliratok
plt.text(min_pts + 11, max_pts - 39, "OVERPERFORMERS", color="white", fontsize=14, fontweight='bold', rotation=45)
plt.text(max_pts - 30, min_pts + 11, "UNDERPERFORMERS", color="white", fontsize=14, fontweight='bold', rotation=45)

# Axis ticks every 10 points
plt.xticks(np.arange(0, max_pts+1, 10), color='white')
plt.yticks(np.arange(0, max_pts+1, 10), color='white')
ax.tick_params(axis='both', which='both', colors='white')

# Axis labels and title
plt.xlabel("Expected Points (Bookmaker)", fontsize=14, color='white')
plt.ylabel("Actual Points", fontsize=14, color='white')
plt.title("Expected vs Actual Points by Season (Premier League)", fontsize=16, fontweight='bold', color='white')

# Legend styling
legend = plt.legend(title="Season", facecolor=background_color, edgecolor='white', labelcolor='white')
plt.setp(legend.get_title(), color='white')

# Spines & grid
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, linestyle='--', alpha=0.3, color='white')

plt.tight_layout()
plt.show()

#%%
# Külön 5-5 lista
top_overall_over = final_df.nlargest(5, 'diff')
top_overall_under = final_df.nsmallest(5, 'diff')

top_overall_over["Label"] = top_overall_over["Team"] + " (" + top_overall_over["SeasonLabel"] + ")"
top_overall_under["Label"] = top_overall_under["Team"] + " (" + top_overall_under["SeasonLabel"] + ")"

# Összefűzés a plothoz
top_combined = pd.concat([top_overall_over, top_overall_under], ignore_index=True)

plt.figure(figsize=(12, 6), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)

# Indexek
x = np.arange(len(top_combined))
bar_width = 0.4

# Oszlopok rajzolása
bars_exp = plt.bar(x - bar_width/2, top_combined["exp_pts"], width=bar_width, color="#4CAF50", label="Expected Points")
bars_act = plt.bar(x + bar_width/2, top_combined["act_pts"], width=bar_width, color="#81C784", label="Actual Points")

# Középső piros vonal
plt.axvline(4.5, color='red', linewidth=2)

# Tengelycímkék
plt.xticks(x, top_combined["Label"], rotation=45, ha="right", color="white", fontsize=10)
plt.yticks(color="white")
ax.tick_params(axis='both', which='both', colors='white')
plt.ylabel("Points", color="white")

# Külön címek bal és jobb oldalra
plt.text(2, max(top_combined["act_pts"]) + 7, "Top 5 Overperformers", ha="center", color="white", fontsize=14, fontweight="bold")
plt.text(7, max(top_combined["act_pts"]) + 7, "Top 5 Underperformers", ha="center", color="white", fontsize=14, fontweight="bold")

# Értékek kiírása az oszlopokra
for bars in [bars_exp, bars_act]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.0f}", ha='center', va='bottom', color="white", fontsize=9)

# Legend
legend = plt.legend(facecolor=background_color, edgecolor='white', labelcolor='white')
plt.setp(legend.get_title(), color='white')

# Stílus
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, linestyle='--', alpha=0.3, color='white')

plt.tight_layout()
plt.show()

#%% Save to Excel
final_df.to_excel("expected_vs_actual.xlsx", index=False)
print("💾 Saved to expected_vs_actual.xlsx")
