import pandas as pd
from fbref import fbref_module as fbref

df, df_year = fbref.get_all_player_data('ESP', year=False)

#%% 
pos = 'FW'
matches_min = 4

df.iloc[:, 7:] = df.iloc[:, 7:].astype(float)

mask = (df.Pos.str.contains(pos)) & (df['Playing Time_90s']>matches_min)
df_filter = df[mask].reset_index(drop=True)

cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'Playing Time_90s']
# Select statistics for comparison
stats_list = ['Per 90 Minutes_G-PK', 'Per 90 Minutes_npxG', 
                 'Per 90 Minutes_Ast', 'Per 90 Minutes_xAG', 
                 'Progression_PrgP', 'Progression_PrgC','Progression_PrgR',
                 'KP', 'Total_Cmp%', 'Tackles_Tkl', 'Int', 'SCA_SCA90']

df_final = df_filter[cols_basic+stats_list]

# per90 values
df_final90 = df_final.copy()
stats_list90 = stats_list.copy()
for i, col in enumerate(stats_list):
    if ('90' in col) or ('%' in col):
        pass
    else:
        df_final90[col] = df_final90[col]/df_final90['Playing Time_90s']
        df_final90.rename(columns={col:f'{col}_p90'}, inplace=True)
        stats_list90[i] = f'{col}_p90'

# Average, deviation and weight for each column
df_avgs = pd.DataFrame(columns=stats_list90)
df_avgs.loc['mean', :] = df_final90[stats_list90].mean()
df_avgs.loc['std', :] = df_final90[stats_list90].std()
## Set importances manually
weights = []
df_avgs.loc['weight', :] = float(1)

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
