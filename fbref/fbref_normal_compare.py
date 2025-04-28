import pandas as pd
from fbref import fbref_module as fbref

df = fbref.get_all_player_data('ESP', year=False)

#%% 
pos = 'MF'
matches_min = 4

# Default dictionary
default = {
    'FW': {
        'Performance_Gls': 1.8,
        'Expected_npxG': 1.5,
        'Standard_SoT': 1.4,
        'Standard_Sh': 1,
        'KP': 1.2,
        'Ast': 1.1,
        'Expected_xAG': 1.3,
        'Take-Ons_Att': 0.9,
        'Take-Ons_Succ': 1.0,
        'Carries_PrgC': 1.0,
        'Carries_1/3': 0.9,
        'Touches_Att 3rd': 0.9,
        'Tackles_Att 3rd': 1.2,
        'Challenges_Tkl': 0.8,
        'Performance_Off': -0.5,
        'Performance_Fls': -0.4
    },
    'MF': {
        'Standard_Sh': 0.7,
        'Total_Cmp': 1.4,
        'Total_Cmp%': 1.3,
        'PrgP': 1.3,
        'KP': 1.2,
        'Ast': 1.1,
        'Expected_xAG': 1.0,
        'Carries_PrgDist': 1.2,
        'Carries_1/3': 1.1,
        'Tackles_Tkl': 1.3,
        'Int': 1.3,
        'Performance_Recov': 1.1,
        'Tackles_Mid 3rd': 1.1,
        'Performance_Fld': 0.9,
        'Performance_CrdY': -0.4
    },
    'DF': {
        'Tackles_Tkl': 1.5,
        'Int': 1.4,
        'Blocks_Blocks': 1.2,
        'Blocks_Pass': 1.1,
        'Clr': 1.3,
        'Aerial Duels_Won%': 1.4,
        'Tackles_Def 3rd': 1.2,
        'Total_Cmp': 1.0,
        'Total_Cmp%': 1.0,
        'Long_Cmp': 0.9,
        'Long_Cmp%': 0.9,
        'Err': -1.0,
        'Performance_OG': -0.7,
        'Performance_CrdY': -0.8,
        'Performance_CrdR': -0.8
    },
    'GK': {
        'Performance_Save%': 1.5,
        'Expected_PSxG+/-': 1.4,
        'Penalty Kicks_Save%': 1.3,
        'Passes_Launch%': 0.5,
        'Total_Cmp': 0.8,
        'Total_Cmp%': 0.8,
        'Performance_CrdR': -0.8
    }
}


mask = (df.Pos.str.contains(pos)) & (df['90s']>matches_min)
df_filter = df[mask].reset_index(drop=True)

cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
# Select statistics for comparison
stats_list = list(default[pos].keys())

df_final = df_filter[cols_basic+stats_list]

# per90 values
df_final90 = df_final.copy()
stats_list90 = stats_list.copy()
for i, col in enumerate(stats_list):
    if ('90' in col) or ('%' in col):
        pass
    else:
        df_final90[col] = df_final90[col]/df_final90['90s']
        df_final90.rename(columns={col:f'{col}_p90'}, inplace=True)
        stats_list90[i] = f'{col}_p90'

# Average, stdeviation and weight for each column
df_avgs = pd.DataFrame(columns=stats_list90)
df_avgs.loc['mean', :] = df_final90[stats_list90].mean()
df_avgs.loc['std', :] = df_final90[stats_list90].std()
## Set importances manually
#weights = []
#df_avgs.loc['weight', :] = float(1)
df_avgs.loc['weight', :] = list(default[pos].values())

#%%
# Normalize
df_final90_normal = df_final90.copy()
for col in stats_list90:
    df_final90_normal[col] = (df_final90[col] - df_avgs.loc['mean', col]) / df_avgs.loc['std', col]

# Calculate scores
for i in range(len(df_final90_normal)):
    multiplies = []
    for x, y in zip(df_final90_normal.loc[i, stats_list90], df_avgs.loc['weight', stats_list90]):
        multiplies.append(x*y)
    df_final90_normal.loc[i, 'score'] = sum(multiplies)
    
# Rename columns
stats_dict = fbref.stats_dict()
for i, col, col90 in zip(range(len(stats_list)), stats_list, stats_list90):
    if col in stats_dict.keys():
        if '_p90' in col90:
            newcol = stats_dict[col]['name'] + ' per 90'
        else:
            newcol = stats_dict[col]['name']
        df_final90_normal.rename(columns={col90: newcol}, inplace=True)

print(df_final90_normal[['Player', 'Squad', 'score']].sort_values(by='score', ascending=False).head(10))
