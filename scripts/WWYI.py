# fetch
import requests
import pandas as pd
from bs4 import BeautifulSoup
from modules import fbref_module as fbr

TEAM_SLUG = "6611f992"  # Ez a rész a Ferencváros egyedi azonosítója
TEAM_NAME = "Ferencvaros"
PLAYER_SLUG = "078a52fe"  # Ez a rész Varga Barnabás egyedi azonosítója
PLAYER_NAME = "Barnabas-Varga"
YEAR = "2024-2025"

# URLs (ezeket lehet majd dinamikusan is cserélni)
TEAM_URL = f"https://fbref.com/en/squads/{TEAM_SLUG}/{YEAR}/matchlogs/all_comps/shooting/{TEAM_NAME}-Match-Logs-All-Competitions"
PLAYER_URL = f"https://fbref.com/en/players/{PLAYER_SLUG}/matchlogs/{YEAR}/{PLAYER_NAME}-Match-Logs"

team_df = fbr.format_column_names(fbr.scrape(TEAM_URL, 'matchlogs_for'))
player_df = fbr.format_column_names(fbr.scrape(PLAYER_URL, 'matchlogs_all'))

team_df = team_df[(team_df.Standard_Gls != 'Standard') & 
                  (team_df.Standard_Gls != 'Gls')].reset_index(drop=True)
player_df = player_df[pd.notna(player_df.Date)].reset_index(drop=True)

# Mentés CSV-be
#team_df.to_csv("ferencvaros_shooting.csv", index=False)
#player_df.to_csv("varga_matchlogs.csv", index=False)

print("Team stats shape:", team_df.shape)
print("Player stats shape:", player_df.shape)

# Első pár sor ellenőrzés
print(team_df.head())
print(player_df.head())

#%% analysis
merged = team_df.merge(
    player_df,
    left_on=[f"For {TEAM_NAME}_Date", f"For {TEAM_NAME}_Opponent"],
    right_on=["Date", "Opponent"],
    how="left",
    suffixes=("_team", "_player")
)

merged["Played"] = merged["Min"].fillna(0).astype(float) >= 30
merged = merged[pd.notna(merged[f"For {TEAM_NAME}_Opponent"])]

# get datatypes right
merged[f"For {TEAM_NAME}_GF"] = merged[f"For {TEAM_NAME}_GF"].str.split("(").str[0].str.strip().astype(float)
merged[f"For {TEAM_NAME}_GA"] = merged[f"For {TEAM_NAME}_GA"].str.split("(").str[0].str.strip().astype(float)
merged["Standard_Sh"] = merged["Standard_Sh"].astype(float)
merged["Standard_G/Sh"] = merged["Standard_G/Sh"].astype(float)
merged["Standard_SoT"] = merged["Standard_SoT"].astype(float)

# calculate other fields
merged["SoT/Sh"] = merged["Standard_SoT"] / merged["Standard_Sh"]
def result_to_points(r):
    if r.startswith("W"): return 3
    elif r.startswith("D"): return 1
    return 0

merged["Points"] = merged[f"For {TEAM_NAME}_Result"].apply(result_to_points)

summary = merged.groupby("Played").agg({
    "Played": "count",
    "For Ferencváros_GF": "mean",
    "For Ferencváros_GA": "mean",
    "Standard_Sh": "mean",
    "Standard_SoT": "mean",
    "Standard_G/Sh": "mean",
    "SoT/Sh": "mean",
    "Points": "mean",
})

#%% viz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import font_manager

# Define consistent colors
colors = {True: 'green', False: 'orangered'}
background_color = '#3c3d3d'
mycolor = '#5ECB43'

# Optional: custom font
my_font_path = os.getcwd()+ r'\Athletic\Nexa-ExtraLight.ttf'
my_font_props = font_manager.FontProperties(fname=my_font_path)

# Rename columns for clarity
renamed = merged.rename(columns={
    f"For {TEAM_NAME}_GF": "Goals For",
    f"For {TEAM_NAME}_GA": "Goals Against",
    "Standard_Sh": "Shots",
    "Standard_SoT": "Shots on Target",
    "SoT/Sh": "SoT per Shot",
    "Standard_G/Sh": "Goals per Shot",
    "Points": "Points"
})

# Set save folder
save_folder = os.getcwd()  # aktuális mappa, szükség esetén cseréld

# 1. Goals For vs Goals Against (barplot average)
summary = renamed.groupby("Played").agg({
    "Goals For": "mean",
    "Goals Against": "mean",
    "Shots": "mean",
    "Shots on Target": "mean",
    "SoT per Shot": "mean",
    "Goals per Shot": "mean",
    "Points": "mean"
}).reset_index()

fig = plt.figure(figsize=(10,6), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)

sum_melt = summary.melt(id_vars="Varga_played", value_vars=["Goals For", "Goals Against"],
                        var_name="Metric", value_name="Value")
sns.barplot(data=sum_melt, x="Varga_played", y="Value", hue="Metric",
            palette=[colors[False], colors[True], colors[False], colors[True]], ax=ax)
ax.set_xticklabels([f"Without {PLAYER_NAME}", f"With {PLAYER_NAME}"], color='white')
ax.set_title(f"{TEAM_NAME} Goals For and Against (per match)", color='white')
ax.set_xlabel("")
ax.set_ylabel("Average per Match", color='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(colors='white')
plt.legend(facecolor=background_color, edgecolor='white', labelcolor='white')

# Add author label
fig.text(0.9, 0.92, 'ADAM JAKUS', color=mycolor, fontsize=16, fontproperties=my_font_props,
         ha='center', va='center')

plt.savefig(f"{save_folder}\\WWYI_{TEAM_NAME}_goals_vs_against.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

# 2. Points per match (barplot)
fig = plt.figure(figsize=(8,5), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)
sns.barplot(data=summary, x="Varga_played", y="Points", palette=[colors[False], colors[True]], ax=ax)
ax.set_xticklabels([f"Without {PLAYER_NAME}", f"With {PLAYER_NAME}"], color='white')
ax.set_title(f"{TEAM_NAME} Points per Match With and Without {PLAYER_NAME}", color='white')
ax.set_xlabel("")
ax.set_ylabel("Average Points per Match", color='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(colors='white')

fig.text(0.9, 0.92, 'ADAM JAKUS', color=mycolor, fontsize=16, fontproperties=my_font_props,
         ha='center', va='center')

plt.savefig(f"{save_folder}\\WWYI_{TEAM_NAME}_points_per_match.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

# 3. Shooting metrics (boxplots)
fig, axes = plt.subplots(1, 2, figsize=(12,5), facecolor=background_color)
for ax in axes:
    ax.set_facecolor(background_color)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white')

sns.boxplot(data=renamed, x="Varga_played", y="Shots", palette=[colors[False], colors[True]], ax=axes[0])
sns.stripplot(data=renamed, x="Varga_played", y="Shots", color="white", alpha=0.6, ax=axes[0])
axes[0].set_xticklabels([f"Without {PLAYER_NAME}", f"With {PLAYER_NAME}"], color='white')
axes[0].set_xlabel("")
axes[0].set_ylabel("Shots", color="white")
axes[0].set_title("Number of Shots per Match", color='white')

sns.boxplot(data=renamed, x="Varga_played", y="SoT per Shot", palette=[colors[False], colors[True]], ax=axes[1])
sns.stripplot(data=renamed, x="Varga_played", y="SoT per Shot", color="white", alpha=0.6, ax=axes[1])
axes[1].set_xticklabels([f"Without {PLAYER_NAME}", f"With {PLAYER_NAME}"], color='white')
axes[1].set_xlabel("")
axes[1].set_ylabel("Percentage of Shots on Target", color="white")
axes[1].set_title("Shot Accuracy (SoT per Shot)", color='white')

plt.suptitle(f"{TEAM_NAME} Shooting Metrics With and Without {PLAYER_NAME}", fontsize=16, color='white')

fig.text(0.9, 0.96, 'ADAM JAKUS', color=mycolor, fontsize=16, fontproperties=my_font_props,
         ha='center', va='center')

plt.tight_layout()
plt.savefig(f"{save_folder}\\WWYI_{TEAM_NAME}_shooting_metrics.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

# 4. Goal Difference over season as scatter plot
renamed = renamed.reset_index().rename(columns={"index": "Match No"})
renamed["Goal Difference"] = renamed["Goals For"] - renamed["Goals Against"]
max_diff = max(abs(renamed["Goal Difference"].max()), abs(renamed["Goal Difference"].min()))

fig = plt.figure(figsize=(12,5), facecolor=background_color)
ax = plt.gca()
ax.set_facecolor(background_color)
colors_list = [colors[x] for x in renamed["Varga_played"]]

for idx, row in renamed.iterrows():
    ax.scatter(row['Match No'], row['Goal Difference'], color=colors_list[idx], s=100)

ax.set_ylim(-max_diff-1, max_diff+1)
ax.set_xlabel("Match Number", color='white')
ax.set_ylabel("Goal Difference", color='white')
ax.set_title(f"{TEAM_NAME} Goal Difference Over the Season (Colored by {PLAYER_NAME} presence)", color='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(colors='white')

fig.text(0.9, -0.003, 'ADAM JAKUS', color=mycolor, fontsize=16, fontproperties=my_font_props,
         ha='center', va='center')

plt.savefig(f"{save_folder}\\WWYI_{TEAM_NAME}_goal_difference_scatter.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
