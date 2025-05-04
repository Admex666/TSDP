import pandas as pd
from fbref import fbref_module as fbref
import numpy as np

df = fbref.get_all_player_data('ESP', year=False)

#%% Determine default weights
pos = 'DF'
role = 'Full-Back'
matches_min = 5
tactic = 'default'

#%% Default dictionary
df['GA/SoTA%'] = np.where(
    (df['Performance_SoTA'].notna()) & (df['Performance_SoTA'] != 0),
    df['Performance_GA'] / df['Performance_SoTA'],
    np.nan
)

# Default dictionary
default = {
    'FW': {
        'Performance_Gls': 1.8,
        'Expected_npxG': 1.5,
        'Standard_SoT': 1.4,
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
        'Performance_Fls': -0.4,
        'Carries_Mis': -0.8,
        'Receiving_Rec': 1.0,
        'Carries_Dis': -0.6
    },
    'MF': {
        'Standard_Sh': 0.7,
        'Total_Cmp': 1.4,
        'Total_Cmp%': 1.3,
        'PrgP': 1.3,
        'KP': 1.2,
        'Ast': 1.1,
        'Expected_xAG': 1.0,
        'SCA_SCA90': 1.0,
        '1/3': 1.0,
        'Carries_PrgDist': 1.2,
        'Carries_1/3': 1.1,
        'Carries_TotDist': 1.2,
        'Tackles_Tkl': 1.3,
        'Int': 1.3,
        'Performance_Recov': 1.1,
        'Tackles_Mid 3rd': 1.1,
        'Performance_Fld': 0.9,
        'Performance_CrdY': -0.4
    },
    'DF': {
        'Tackles_Tkl': 1.5,
        'Challenges_Tkl%': 1.3,
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
        'GA/SoTA%': 1.4,
        'Expected_PSxG+/-': 1.4,
        'Penalty Kicks_Save%': 1.3,
        'Passes_Launch%': 0.5,
        'Total_Cmp': 0.8,
        'Total_Cmp%': 0.8,
        'Performance_CrdR': -0.8,
        'Performance_CS%': 1.2,
        'Sweeper_#OPA/90': 0.8,
        'Crosses_Stp%': 1.1
    }
}


#%% Positional weights
position_roles = {
    'DF': ['Central Defender', 'Ball Playing Defender', 'Libero', 
           'Full-Back', 'Wing-Back', 'Inverted Wing-Back'],
    'MF': ['Defensive Midfielder', 'Ball Winning Midfielder', 'Deep Lying Playmaker',
           'Box to Box Midfielder', 'Mezzala', 'Central Midfielder', 
           'Advanced Playmaker', 'Shadow Striker', 'Attacking Midfielder'],
    'FW': ['Advanced Forward', 'Complete Forward', 'Poacher', 'Inside Forward', 'Target Man'],
    'GK': ['Goalkeeper', 'Sweeper Keeper']
}

positional_weights = {}
for p in position_roles.keys():
    for r in position_roles[p]:
        positional_weights[r] = default[p].copy()

# Ball Playing Defender – magasabb passz-, technika- és vonalbontó képesség:
positional_weights['Ball Playing Defender'].update({
    'Total_Cmp': 1.3,
    'Total_Cmp%': 1.1,
    'Long_Cmp%': 1.2,
    'Carries_PrgDist': 1.3,
    'Carries_Carries': 1.2,
    'Sweeper_#OPA/90': 1.1,
    'Err': -1.2
})

# Deep Lying Playmaker – passzátlagok, kreativitás és labdabiztonság:
positional_weights['Deep Lying Playmaker'].update({
    'Total_Cmp%': 1.5,
    'PrgP': 1.4,
    'KP': 1.3,
    'SCA_SCA90': 1.2,
    'Touches_Touches': 1.2,
    'Carries_TotDist': 1.1,
    'Performance_CrdY': -0.6
})

# Ball Winning Midfielder – szerelések, labdaszerzés és fizikai aktivitás:
positional_weights['Ball Winning Midfielder'].update({
    'Tackles_Tkl': 1.6,
    'Challenges_Tkl%': 1.5,
    'Performance_Int': 1.4,
    'Touches_Live': 1.2,
    'Performance_Fld': 1.0
})

# Inside Forward – befejezés, 1-az-1, kreatív passzok:
positional_weights['Inside Forward'].update({
    'Performance_Gls': 1.7,
    'Expected_npxG': 1.6,
    'Take-Ons_Succ': 1.2,
    'Standard_SoT': 1.3,
    'Ast': 1.1,
    'Receiving_Rec': 1.0
})

# Poacher – kizárólag befejezés és helyzetkihasználás:
positional_weights['Poacher'].update({
    'Performance_Gls': 2.0,
    'Expected_xG': 1.8,
    'Expected_npxG': 1.8,
    'Standard_SoT': 1.5,
    'Carries_PrgC': 0.6,
    'Tackles_Att 3rd': 0.3
})

# Sweeper Keeper – mentések, sweeper-akciók és vonalbontó passzok:
positional_weights['Sweeper Keeper'].update({
    'Performance_Save%': 1.6,
    'Expected_PSxG+/-': 1.5,
    'Sweeper_#OPA/90': 1.3,
    'Crosses_Stp%': 1.2,
    'GA/SoTA%': 1.4,
    'Passes_Launch%': 1.1,
    'Total_Cmp%': 0.9
})

# Central Defender – hagyományos középső védő, a szerelés és blokkolás dominál:
positional_weights['Central Defender'].update({
    'Tackles_Tkl': 1.6,
    'Blocks_Blocks': 1.4,
    'Blocks_Pass': 1.3,
    'Clr': 1.4,
    'Aerial Duels_Won%': 1.5,
    'Performance_OG': -0.9,
    'Err': -1.2
})

# Libero – vonalbontó, technikai védő aki beljebb jön:
positional_weights['Libero'].update({
    'Total_Cmp': 1.3,
    'PrgP': 1.2,
    'Carries_PrgDist': 1.2,
    'Sweeper_#OPA/90': 1.2,
    'Touches_Def 3rd': 1.1,
    'Err': -1.0
})

# Full-Back – klasszikus szélső védő, támogató passzok és blokkok:
positional_weights['Full-Back'].update({
    'Performance_Crs': 1.2,
    'CrsPA': 1.1,
    'Total_Cmp%': 1.1,
    'Short_Cmp%': 1.1,
    'Tackles_Tkl': 1.3,
    'Performance_Int': 1.2
})

# Wing-Back – támadóbb full-back, sok beívelés és előrevitele:
positional_weights['Wing-Back'].update({
    'Performance_Crs': 1.3,
    'CrsPA': 1.2,
    'Touches_Att 3rd': 1.2,
    'Carries_PrgC': 1.1,
    'Tackles_Tkl': 1.2,
    'Performance_Recov': 1.1
})

# Inverted Wing-Back – beljebb húzódó védő, passzjáték és progresszió:
positional_weights['Inverted Wing-Back'].update({
    'Total_Cmp%': 1.2,
    'PrgP': 1.1,
    'Carries_PrgDist': 1.1,
    'Tackles_Def 3rd': 1.2,
    'Aerial Duels_Won%': 1.1
})

# Defensive Midfielder – elsődleges védekezés, szerelések, labdaszerzés:
positional_weights['Defensive Midfielder'].update({
    'Tackles_Tkl': 1.6,
    'Challenges_Tkl%': 1.4,
    'Performance_Int': 1.4,
    'Performance_Recov': 1.3,
    'Touches_Def 3rd': 1.1
})

# Box to Box Midfielder – sokoldalú, támadás és védekezés is fontos:
positional_weights['Box to Box Midfielder'].update({
    'Tackles_Tkl': 1.4,
    'Performance_Recov': 1.2,
    'Carries_PrgDist': 1.2,
    'PrgP': 1.2,
    'Ast': 1.1,
    'Performance_Gls': 1.1
})

# Mezzala – belső támadó középpályás, kreatív és progresszív passzok:
positional_weights['Mezzala'].update({
    'PrgP': 1.4,
    'KP': 1.3,
    'Carries_PrgDist': 1.2,
    'SCA_SCA90': 1.2,
    'Touches_Mid 3rd': 1.1
})

# Central Midfielder – kiegyensúlyozott box-to-box/support szerep:
positional_weights['Central Midfielder'].update({
    'Total_Cmp%': 1.3,
    'Carries_TotDist': 1.2,
    'PrgP': 1.2,
    'Ast': 1.0,
    'Performance_Fld': 1.0
})

# Advanced Playmaker – kreatív szervező, passzpontosság és helyzetkeltés:
positional_weights['Advanced Playmaker'].update({
    'Total_Cmp%': 1.5,
    'PrgP': 1.4,
    'KP': 1.4,
    'Expected_xAG': 1.3,
    'SCA_SCA90': 1.3,
    'Performance_CrdY': -0.5
})

# Shadow Striker – mögöttes csatár, gólérzékeny és helyzetkihasználó:
positional_weights['Shadow Striker'].update({
    'Performance_Gls': 1.7,
    'Expected_npxG': 1.6,
    'Standard_SoT': 1.4,
    'SCA_SCA90': 1.2,
    'Ast': 1.1,
    'Tackles_Att 3rd': 1.0
})

# Attacking Midfielder – középcsatár-szervező, passzok és lövések:
positional_weights['Attacking Midfielder'].update({
    'KP': 1.3,
    'Expected_xAG': 1.2,
    'Performance_Ast': 1.2,
    'Performance_Gls': 1.2,
    'PrgP': 1.1
})

# Advanced Forward – gyors, direkt, mélységi indításokra optimalizált:
positional_weights['Advanced Forward'].update({
    'Performance_Gls': 1.8,
    'Expected_npxG': 1.6,
    'Standard_SoT': 1.5,
    'Take-Ons_Att': 1.1,
    'Take-Ons_Succ': 1.2,
    'Carries_PrgC': 1.1,
    'Tackles_Att 3rd': 0.8
})

# Complete Forward – minden elemben (térkihasználás, tartás, kreativitás) jól szerepel:
positional_weights['Complete Forward'].update({
    'Performance_Gls': 1.6,
    'Expected_xG': 1.5,
    'Expected_xAG': 1.4,
    'KP': 1.3,
    'Ast': 1.2,
    'Carries_PrgC': 1.2,
    'Carries_1/3': 1.1,
    'Tackles_Att 3rd': 1.0,
    'Challenges_Tkl': 1.0
})

# Target Man – erős testfelépítés, letámadás, holt labdás játékok:
positional_weights['Target Man'].update({
    'Aerial Duels_Won%': 2.0,
    'Aerial Duels_Won': 1.7,
    'Performance_Gls': 1.2,
    'Standard_SoT': 1.1,
    'Carries_Carries': 1.3,
    'Carries_TotDist': 1.2,
    'Performance_Crs': 1.0,
    'Performance_Fls': -0.2
})


#%% Tactical weights
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

#%% Filter df and Calculate avg, std, weights
if role == '':
    weight_dict = tweights[tactic][pos].copy()
elif role == 'default':
    weight_dict = default[pos].copy()
else:
    weight_dict = positional_weights[role].copy()
    
mask = (df.Pos.str.contains(pos)) & (df['90s']>matches_min)
df_filter = df[mask].reset_index(drop=True)

cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
# Select statistics for comparison
stats_list = list(weight_dict.keys())

df_final = df_filter[cols_basic+stats_list]
df_final.fillna(0, inplace=True)

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
df_avgs.loc['weight', :] = list(weight_dict.values())
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
df_final90_normal['index'] = round((df_final90_normal['score'] / (weights_sum*1.5)*100), 1)

print(df_final90_normal[['Pos', 'Player', 'Squad', 'score', 'index']].sort_values(by='score', ascending=False).head(10))

#%% Compare each tactic to default
for t in list(tweights.keys())[1:]:
    print(f'{t} \n')
    for p in tweights['default'].keys():
        for s in tweights['default'][p].keys():
            w_def = tweights['default'][p][s]
            w_tactic = tweights[t][p][s]
            diff = int( (w_tactic / w_def -1) *100 )
            print(f'{s} diff.: {diff}%')
            
            
#%% Correlations
df_corr_input = df_final[stats_list]
correlation_matrix = df_corr_input.corr(method='pearson')

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()