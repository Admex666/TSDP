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
points_dict[team][points_dict[team]['Week'] == 15]['PointsFor_cumsum'].iloc[0]

# Divide into groups
df["Period"] = pd.cut(df["Week"], bins=[0, 11, 25, 36], labels=["Early", "Mid", "Late"])
print(df.groupby("Period")[["TotalGoals", "TotalXG"]].mean())

# Boxplot: goals
sns.boxplot(data=df, x="Period", y="TotalGoals")
plt.title("Distribution of Goals in Periods")
plt.show()

# Boxplot: xG
sns.boxplot(data=df, x="Period", y="TotalXG")
plt.title("Distribution of Expected Goals in Periods")
plt.show()

#%% Test significance
early = df[df["Period"] == "Early"]["TotalGoals"]
late = df[df["Period"] == "Late"]["TotalGoals"]

t_stat, p_val = ttest_ind(early, late, equal_var=False)
print(f"T-test on goals (Early vs Late): t={t_stat:.3f}, p={p_val:.3f}")

#%% Trend analysis
slope, intercept, r_value, p_value, std_err = linregress(x=df["Week"], y=df["TotalGoals"])
print(f"Goal trend regression: slope={slope:.3f}, p={p_value:.3f}, R2={r_value**2}")

# Ábra a trendről
sns.regplot(data=df, x="Week", y="TotalGoals", scatter_kws={"alpha":0.4})
plt.title("Trend of goals during the season")
plt.show()

#%% Difference in xG: do teams get closer or further
df["XG_Diff"] = abs(df["HomeXG"] - df["AwayXG"])
sns.lineplot(x="Week", y="XG_Diff", data=df)
sns.regplot(data=df, x="Week", y="XG_Diff", scatter_kws={"alpha":0.4})
plt.ylim(top=3)
plt.title('Difference of Expected Goals in a match, during the season')
plt.show()
