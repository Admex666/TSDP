import pandas as pd
from fbref import fbref_module as fbref

df = fbref.get_all_player_data('ESP', year=False)

#%% Determine weights
pos = 'MF'
matches_min = 5
tactic = 'low_block'

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

tweights = {
    'default': default,

    'gegenpress': {
        'FW': {
            'Performance_Gls': 1.5,
            'Expected_npxG': 1.4,
            'Standard_SoT': 1.3,
            'Standard_Sh': 1.0,
            'KP': 1.2,
            'Ast': 1.1,
            'Expected_xAG': 1.2,
            'Take-Ons_Att': 1.0,
            'Take-Ons_Succ': 1.1,
            'Carries_PrgC': 1.1,
            'Carries_1/3': 1.0,
            'Touches_Att 3rd': 1.1,
            'Tackles_Att 3rd': 1.5,
            'Challenges_Tkl': 1.2,
            'Performance_Off': -0.5,
            'Performance_Fls': -0.4
        },
        'MF': {
            'Standard_Sh': 0.8,
            'Total_Cmp': 1.3,
            'Total_Cmp%': 1.2,
            'PrgP': 1.4,
            'KP': 1.3,
            'Ast': 1.2,
            'Expected_xAG': 1.1,
            'Carries_PrgDist': 1.3,
            'Carries_1/3': 1.2,
            'Tackles_Tkl': 1.5,
            'Int': 1.4,
            'Performance_Recov': 1.2,
            'Tackles_Mid 3rd': 1.3,
            'Performance_Fld': 1.0,
            'Performance_CrdY': -0.4
        },
        'DF': {
            'Tackles_Tkl': 1.6,
            'Int': 1.5,
            'Blocks_Blocks': 1.3,
            'Blocks_Pass': 1.2,
            'Clr': 1.2,
            'Aerial Duels_Won%': 1.3,
            'Tackles_Def 3rd': 1.4,
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
            'Performance_Save%': 1.6,
            'Expected_PSxG+/-': 1.5,
            'Penalty Kicks_Save%': 1.4,
            'Passes_Launch%': 0.6,
            'Total_Cmp': 0.9,
            'Total_Cmp%': 0.9,
            'Performance_CrdR': -0.8
        }
    },

    'possession': {
        'FW': {
            'Performance_Gls': 1.6,
            'Expected_npxG': 1.5,
            'Standard_SoT': 1.4,
            'Standard_Sh': 1.0,
            'KP': 1.3,
            'Ast': 1.2,
            'Expected_xAG': 1.3,
            'Take-Ons_Att': 1.0,
            'Take-Ons_Succ': 1.1,
            'Carries_PrgC': 1.2,
            'Carries_1/3': 1.1,
            'Touches_Att 3rd': 1.2,
            'Tackles_Att 3rd': 0.8,
            'Challenges_Tkl': 0.7,
            'Performance_Off': -0.4,
            'Performance_Fls': -0.3
        },
        'MF': {
            'Standard_Sh': 0.7,
            'Total_Cmp': 1.5,
            'Total_Cmp%': 1.4,
            'PrgP': 1.5,
            'KP': 1.4,
            'Ast': 1.3,
            'Expected_xAG': 1.2,
            'Carries_PrgDist': 1.4,
            'Carries_1/3': 1.3,
            'Tackles_Tkl': 1.0,
            'Int': 1.0,
            'Performance_Recov': 1.1,
            'Tackles_Mid 3rd': 1.0,
            'Performance_Fld': 0.9,
            'Performance_CrdY': -0.4
        },
        'DF': {
            'Tackles_Tkl': 1.2,
            'Int': 1.1,
            'Blocks_Blocks': 1.0,
            'Blocks_Pass': 0.9,
            'Clr': 1.0,
            'Aerial Duels_Won%': 1.1,
            'Tackles_Def 3rd': 1.1,
            'Total_Cmp': 1.2,
            'Total_Cmp%': 1.2,
            'Long_Cmp': 1.0,
            'Long_Cmp%': 1.0,
            'Err': -1.0,
            'Performance_OG': -0.7,
            'Performance_CrdY': -0.8,
            'Performance_CrdR': -0.8
        },
        'GK': {
            'Performance_Save%': 1.4,
            'Expected_PSxG+/-': 1.3,
            'Penalty Kicks_Save%': 1.2,
            'Passes_Launch%': 0.4,
            'Total_Cmp': 1.0,
            'Total_Cmp%': 1.0,
            'Performance_CrdR': -0.8
        }
    },

    'low_block': {
        'FW': {
            'Performance_Gls': 1.7,
            'Expected_npxG': 1.6,
            'Standard_SoT': 1.5,
            'Standard_Sh': 1.0,
            'KP': 1.1,
            'Ast': 1.0,
            'Expected_xAG': 1.0,
            'Take-Ons_Att': 0.9,
            'Take-Ons_Succ': 1.0,
            'Carries_PrgC': 0.9,
            'Carries_1/3': 0.8,
            'Touches_Att 3rd': 0.8,
            'Tackles_Att 3rd': 0.7,
            'Challenges_Tkl': 0.6,
            'Performance_Off': -0.4,
            'Performance_Fls': -0.3
        },
        'MF': {
            'Standard_Sh': 0.6,
            'Total_Cmp': 1.2,
            'Total_Cmp%': 1.1,
            'PrgP': 1.2,
            'KP': 1.1,
            'Ast': 1.0,
            'Expected_xAG': 0.9,
            'Carries_PrgDist': 1.1,
            'Carries_1/3': 1.0,
            'Tackles_Tkl': 1.4,
            'Int': 1.3,
            'Performance_Recov': 1.2,
            'Tackles_Mid 3rd': 1.3,
            'Performance_Fld': 0.8,
            'Performance_CrdY': -0.4
        },
        'DF': {
            'Tackles_Tkl': 1.7,
            'Int': 1.6,
            'Blocks_Blocks': 1.4,
            'Blocks_Pass': 1.3,
            'Clr': 1.5,
            'Aerial Duels_Won%': 1.6,
            'Tackles_Def 3rd': 1.5,
            'Total_Cmp': 0.9,
            'Total_Cmp%': 0.9,
            'Long_Cmp': 0.8,
            'Long_Cmp%': 0.8,
            'Err': -1.0,
            'Performance_OG': -0.7,
            'Performance_CrdY': -0.8,
            'Performance_CrdR': -0.8
        },
        'GK': {
            'Performance_Save%': 1.7,
            'Expected_PSxG+/-': 1.6,
            'Penalty Kicks_Save%': 1.5,
            'Passes_Launch%': 0.5,
            'Total_Cmp': 0.8,
            'Total_Cmp%': 0.8,
            'Performance_CrdR': -0.8
        }
    }
}


mask = (df.Pos.str.contains(pos)) & (df['90s']>matches_min)
df_filter = df[mask].reset_index(drop=True)

cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
# Select statistics for comparison
stats_list = list(tweights[tactic][pos].keys())

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
df_avgs.loc['weight', :] = list(tweights[tactic][pos].values())
weights_sum = df_avgs.loc['weight', :].sum()

#%% Normalize, calculate index
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
df_final90_normal['index'] = (df_final90_normal['score'] / (weights_sum*1.5)*100).astype(int)

print(df_final90_normal[['Player', 'Squad', 'score', 'index']].sort_values(by='score', ascending=False).head(10))

#%% Compare each tactic to default
for t in list(tweights.keys())[1:]:
    print(f'{t} \n')
    for p in tweights['default'].keys():
        for s in tweights['default'][p].keys():
            w_def = tweights['default'][p][s]
            w_tactic = tweights[t][p][s]
            diff = int( (w_tactic / w_def -1) *100 )
            print(f'{s} diff.: {diff}%')