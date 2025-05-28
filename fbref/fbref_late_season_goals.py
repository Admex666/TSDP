#%% Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, linregress
from fbref import fbref_module as fbr

#%% get data
url = 'https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures'
table_id = 'sched_2024-2025_11_1'
df_raw = fbr.scrape(url,table_id)

mask_none = (df_raw.Wk != 'Wk') & (pd.notna(df_raw.Wk))
df_raw = df_raw[mask_none]
df_raw[['HomeGoals', 'AwayGoals']] = df_raw.Score.str.split('–', expand=True)

#%% Only relevant data
# Format
df = df_raw[['Wk', 'Home', 'HomeGoals', 'xG', 'Away', 'AwayGoals', 'xG.1']].copy()
df.rename(columns={'xG': 'HomeXG', 'xG.1': 'AwayXG', 'Wk': 'Week'}, inplace=True)
df[['Week', 'HomeGoals', 'AwayGoals', 'HomeXG', 'AwayXG']] = df[['Week', 'HomeGoals', 'AwayGoals', 'HomeXG', 'AwayXG']].astype(float)
df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]
df["TotalXG"] = df["HomeXG"] + df["AwayXG"]

df = df[df.Week <= 36]
# Divide into groups
df["Period"] = pd.cut(df["Week"], bins=[0, 11, 25, 36], labels=["Early", "Mid", "Late"])
print(df.groupby("Period")[["TotalGoals", "TotalXG"]].mean())

# Summarize points for each team
points_dict = {}
for team in df.Home.unique():
    points_dict[team] = []
for team in points_dict.keys():
    df_team = df[(df.Home == team) | (df.Away == team)]
    df_team['isHome'] = df_team.Home == team
    df_team['GoalsFor'] = np.where(df_team.isHome, df_team.HomeGoals, df_team.AwayGoals)
    df_team['GoalsAgainst'] = np.where(df_team.isHome, df_team.AwayGoals, df_team.HomeGoals)
    df_team['PointsFor'] = np.where(df_team.GoalsFor > df_team.GoalsAgainst, 3,
                                    np.where(df_team.GoalsFor == df_team.GoalsAgainst, 1,
                                             0)
                                    )
    for x in ['GoalsFor', 'GoalsAgainst', 'PointsFor']:
        df_team[f'{x}_cumsum'] = df_team[f'{x}'].cumsum()
    
    points_dict[team] = df_team

# Pair points for each round (df)
for i, row in df.iterrows():
    home = row['Home']
    away = row['Away']
    week = row['Week']
    # data for home
    row_week_home = points_dict[home][points_dict[home]['Week'] == week]
    points_home = row_week_home['PointsFor_cumsum'].iloc[0]
    gf_home = row_week_home['GoalsFor_cumsum'].iloc[0]
    ga_home = row_week_home['GoalsAgainst_cumsum'].iloc[0]
    # for away
    row_week_away = points_dict[away][points_dict[away]['Week'] == week]
    points_away = row_week_away['PointsFor_cumsum'].iloc[0]
    gf_away = row_week_away['GoalsFor_cumsum'].iloc[0]
    ga_away = row_week_away['GoalsAgainst_cumsum'].iloc[0]
    
    # add to df
    df.loc[i, ['HomePoints', 'HomeGF', 'HomeGA']] = points_home, gf_home, ga_home
    df.loc[i, ['AwayPoints', 'AwayGF', 'AwayGA']] = points_away, gf_away, ga_away

#%% Monte carlo
games_per_season = 38
europe_threshold = 62
relegation_threshold = 33
simulations = 1000

def simulate_finish(current_points, matches_remaining, simulations=10000):
    matches_remaining = int(matches_remaining)
    results = []
    for _ in range(simulations):
        simulated_results = np.random.choice([0, 1, 3], size=matches_remaining, p=[0.4, 0.3, 0.3])
        results.append(current_points + simulated_results.sum())
    return results


def probability_to_reach(points_list, target):
    return np.mean(np.array(points_list) >= target)

def compute_probabilities(row):
    home_remaining = games_per_season - row["Week"]
    away_remaining = games_per_season - row["Week"]

    home_sim = simulate_finish(row["HomePoints"], home_remaining, simulations)
    away_sim = simulate_finish(row["AwayPoints"], away_remaining, simulations)

    home_prob_eur = probability_to_reach(home_sim, europe_threshold)
    home_prob_releg = 1 - probability_to_reach(home_sim, relegation_threshold)

    away_prob_eur = probability_to_reach(away_sim, europe_threshold)
    away_prob_releg = 1 - probability_to_reach(away_sim, relegation_threshold)

    return pd.Series({
        "HomeProbEur": home_prob_eur,
        "HomeProbReleg": home_prob_releg,
        "AwayProbEur": away_prob_eur,
        "AwayProbReleg": away_prob_releg
    })

df[["HomeProbEur", "HomeProbReleg", "AwayProbEur", "AwayProbReleg"]] = df.apply(compute_probabilities, axis=1)

def classify_probabilistic_tension(row, threshold=0.3):
    if (row["HomeProbEur"] < threshold and row["HomeProbReleg"] < threshold and
        row["AwayProbEur"] < threshold and row["AwayProbReleg"] < threshold):
        return "No realistic tension"
    else:
        return "Tension possible"

df["ProbabilisticTension"] = df.apply(classify_probabilistic_tension, axis=1)

#%% Boxplots
# xG
sns.boxplot(data=df, x="Period", y="TotalXG")
plt.title("Distribution of Expected Goals in Periods")
plt.show()

# Goals
sns.boxplot(data=df, x="Period", y="TotalGoals")
plt.title("Distribution of Goals, Serie A 24/25")
plt.show()
#plt.savefig(r'C:\Users\Adam\Dropbox\TSDP_output\fbref\2025.05\2025.05.21., LSE, Serie A.png',dpi=300, bbox_inches='tight')

#%% Test significance of season period
early = df[df["Period"] == "Early"]["TotalGoals"]
late = df[df["Period"] == "Late"]["TotalGoals"]

t_stat, p_val = ttest_ind(early, late, equal_var=False)
print(f"T-test on goals (Early vs Late): t={t_stat:.3f}, p={p_val:.3f}")

#%% Test significance of tension
tension0 = df[df["ProbabilisticTension"] == "No realistic tension"]["TotalGoals"]
tension1 = df[df["ProbabilisticTension"] == "Tension possible"]["TotalGoals"]
t_stat, p_val = ttest_ind(tension0, tension1, equal_var=False)
print(f"T-test on goals (No tension for one vs Tension possible): t={t_stat:.3f}, p={p_val:.3f}")

# viz
bar_t0 = df[df.ProbabilisticTension == "No realistic tension"].groupby('TotalGoals').TotalGoals.count()
bar_t1 = df[df.ProbabilisticTension == "Tension possible"].groupby('TotalGoals').TotalGoals.count()

# Az összes lehetséges x érték, hogy minden oszlopra jusson
all_goals = sorted(set(bar_t0.index).union(set(bar_t1.index)))

bar_width = 0.4

# X pozíciók az eltoláshoz
x = np.array(all_goals)
x0 = x - bar_width / 2
x1 = x + bar_width / 2

# Heights
y0 = [bar_t0.get(val, 0) for val in x]
y1 = [bar_t1.get(val, 0) for val in x]

# Plot
plt.bar(x0, y0, width=bar_width, label="No realistic tension", color="skyblue", edgecolor="black")
plt.bar(x1, y1, width=bar_width, label="Tension possible", color="salmon", edgecolor="black")

# Numbers outwritten
for xi, yi in zip(x0, y0):
    rel0 = int(yi/sum(bar_t0)*100)
    plt.text(xi, yi + 0.3, f'{rel0}%', ha='center', va='bottom', fontsize=8)

for xi, yi in zip(x1, y1):
    rel1 = int(yi/sum(bar_t1)*100)
    plt.text(xi, yi + 0.3, f'{rel1}%', ha='center', va='bottom', fontsize=8)
    
# Legend, title
plt.title("Goals in Matches With or Without Probabilistic Tension")
plt.xlabel("Total Goals")
plt.ylabel("Match Count")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.show()

#%% Trend analysis
slope, intercept, r_value, p_value, std_err = linregress(x=df["Week"], y=df["TotalGoals"])
print(f"Goal trend regression: slope={slope:.3f}, p={p_value:.3f}, R2={r_value**2}")

# Plot trend
sns.regplot(data=df, x="Week", y="TotalGoals", scatter_kws={"alpha":0.4})
plt.title("Trend of goals during the season")
plt.show()

#%% Difference in xG: do teams get closer or further
df["XG_Diff"] = abs(df["HomeXG"] - df["AwayXG"])
slope, intercept, r_value, p_value, std_err = linregress(x=df["Week"], y=df["XG_Diff"])
print(f"Goal trend regression: slope={slope:.3f}, p={p_value:.3f}, R2={r_value**2}")
#sns.lineplot(x="Week", y="XG_Diff", data=df)
sns.regplot(data=df, x="Week", y="XG_Diff", scatter_kws={"alpha":0.4})
plt.ylim(top=3)
plt.title('Difference of Expected Goals in a match, during the season')
plt.show()

