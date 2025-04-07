# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:24:05 2025

@author: Adam
"""
# Set and scrape league dataframes
import pandas as pd
from fbref import fbref_module as fbref
from bs4 import BeautifulSoup
import requests

league = 'FRA'
URL_match = 'https://fbref.com/en/matches/37a1d2be/Marseille-Toulouse-April-6-2025-Ligue-1'

comp_id, league_name = fbref.team_dict_get(league)

#URLs
URL_standard = f"https://fbref.com/en/comps/{comp_id}/stats/{league_name}-Stats#all_stats_standard"

stats_list = ['passing', 'possession', 'shooting', 'defense', 'misc']
for stat in stats_list:
    globals()[f'URL_{stat}'] = f"https://fbref.com/en/comps/{comp_id}/{stat}/{league_name}-Stats#all_stats_{stat}" 

#%% Scrape the competition stat dfs
stats_list.append('standard')
for stat in stats_list:
    globals()[f'df_{stat}'] = fbref.format_column_names(fbref.scrape(globals()[f'URL_{stat}'], f'stats_{stat}'))
    # Cleaning header rows
    globals()[f'df_{stat}'].drop(globals()[f'df_{stat}'][globals()[f'df_{stat}']['Rk']=='Rk'].index, inplace=True)


#%% Scrape a specific match
match_stats_list = ['summary', 'passing', 'defense', 'possession', 'misc']

# Get the team ids
response = requests.get(URL_match)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table with team links
teamlogos = soup.find_all(class_='teamlogo')
teams = []
for logo in teamlogos:
    src = logo.get("src")
    alt = logo.get("alt")
    teams.append({'src': src,
                  'alt': alt})
    
teams_df = pd.DataFrame(teams).drop_duplicates()
teams_df['team_name'] = teams_df.alt.str.replace('Club Crest', '').str.strip()
teams_df['team_id'] = teams_df.src.str.split('/').str.get(-1).str.split('.png').str.get(0)

# with the team_ids: scrape tables from fbref
for i, team in teams_df.iterrows():
    team_id = team.team_id
    for stat in match_stats_list:
        globals()[f'df_{i}_{stat}'] = fbref.format_column_names(fbref.scrape(URL_match, f'stats_{team_id}_{stat}'))
        globals()[f'df_{i}_{stat}']['Squad'] = team.team_name

#%%
cols_basic = ['Player','Squad','#','Nation','Pos','Age']
cols_to_end = list(df_1_summary.columns[5:-1])
col_order = cols_basic + cols_to_end
df_0_summary = df_0_summary.loc[:,col_order]
df_1_summary = df_1_summary.loc[:,col_order]

#%%
# Create a merged season df
df_merged = df_passing.copy()

for stat in stats_list[1:]:
    df_merged = pd.merge(df_merged, globals()[f'df_{stat}'],
                         how='inner', on=['Player', 'Squad'], 
                         suffixes=('', '_remove'))
    # Remove the duplicate columns
    df_merged.drop([i for i in df_merged.columns if 'remove' in i],
                   axis=1, inplace=True)
    
# Create merged match dfs for each team
for team in [0, 1]: 
    globals()[f'df_merged{team}'] = globals()[f'df_{team}_summary'].copy()
    
    for stat in match_stats_list[1:]:
        globals()[f'df_merged{team}'] = pd.merge(globals()[f'df_merged{team}'],
                                                 globals()[f'df_{team}_{stat}'],
                                                 how='inner', on='Player', 
                                                 suffixes=('', '_remove'))
        globals()[f'df_merged{team}'].drop([i for i in globals()[f'df_merged{team}'].columns if 'remove' in i],
                       axis=1, inplace=True) ## Remove the duplicate columns
    
# Find the column names that are missing or not named the same
missing_nr = 0
missing_cols = []
for col in df_merged1.columns.unique():
    if col not in df_merged.columns.unique():
        missing_nr += 1
        missing_cols.append(col)
print(f'{missing_nr} cols missing:\n{missing_cols}')

#%% Find and rename the columns in merged season df
df_merged.rename(columns={'Playing Time_Min': 'Min',
                          'Standard_Sh':'Performance_Sh',
                          'Standard_SoT': 'Performance_SoT',
                          },
                 inplace=True)
df_merged = df_merged.dropna(subset=['Pos']).reset_index(drop=True)

df_merged_match = pd.concat([df_merged0, df_merged1]).reset_index(drop=True)
df_merged_match.drop(columns=missing_cols[4:]+['#'],
                                   inplace=True)
# and remove unnecessary summarizing row
index_to_drop = df_merged_match[df_merged_match.Player.str.contains('Player')].index
df_merged_match.drop(index=index_to_drop, inplace=True)
df_merged_match = df_merged_match.reset_index(drop=True)

df_merged = df_merged.loc[:,df_merged_match.columns.unique()]
df_merged.iloc[:,5:] = df_merged.iloc[:,5:].astype(float)

#%% Get league per90 values
df_merged_p90 = df_merged.copy()
league_avg_dict = {}
stat_cols = df_merged_p90.columns[6:]
for col in stat_cols:
    if ('%' in col) or ('90' in col):
        pass
    else:
        df_merged_p90[col] = df_merged_p90[col]/df_merged_p90.Min*90
    league_avg_dict[col] = df_merged_p90[col].mean()

# Get per90 averages for positions as well
# df_merged_match.Pos.unique()
df_merged_match.Pos = df_merged_match.Pos.str.split(',').str[0].str.strip()
pos_dict = {'GK': ['GK'],
            'DF': ['CB', 'RB', 'LB', 'WB'],
            'MF': ['DM', 'RM', 'CM', 'LM', 'AM'],
            'FW': ['FW', 'LW', 'RW']
            }
wing_dict = {'Winger':['RB','LB','RM','LM','FW','LW', 'RW', 'WB'],
             'Not Winger':['GK','CB','DM','CM','AM','FW']
             }

df_merged_match['pos_group'] = None
df_merged_match['winger'] = None
for posnr in range(len(df_merged_match.Pos)):
    position = df_merged_match.Pos[posnr]
    
    for kp in pos_dict.keys():
        poslist = pos_dict.get(kp)
        for lp in range(len(poslist)):
            if poslist[lp] in position:
                df_merged_match['pos_group'][posnr] = kp
    for kw in wing_dict.keys():
        wlist =  wing_dict.get(kw)
        for lw in range(len(wlist)):
            if wlist[lw] in position:
                df_merged_match['winger'][posnr] = kw

# Get position group averages in league
pos_avg_dict = {'GK':{}, 'DF':{}, 'MF':{}, 'FW': {}}
for posgr in pos_dict.keys():
    globals()[f'df_{posgr}'] = df_merged_p90[df_merged_p90.Pos.str.contains(posgr)]
    tempdict = {}
    for col in stat_cols:
        tempdict[col] = globals()[f'df_{posgr}'][col].mean()
    pos_avg_dict[posgr] = tempdict

#%% Create dataframes that include all stats
# Only want relevant squads' season data
for suffix in ['', '_p90']:
    globals()[f'df_merged{suffix}'].Squad = globals()[f'df_merged{suffix}'].Squad.replace(to_replace=['it Inter', 'nl Feyenoord'], value=['Internazionale', 'Feyenoord'])

if (teams_df.team_name[0] in df_merged.Squad.unique()) & (teams_df.team_name[1] in df_merged.Squad.unique()):
    df_merged_sq = df_merged[df_merged.Squad.isin([teams_df.team_name[0], teams_df.team_name[1]])]
    df_merged_p90_sq = df_merged_p90[df_merged_p90.Squad.isin([teams_df.team_name[0], teams_df.team_name[1]])]
    print('Team names found')
else:
    print('Error: team names not found')

#%%
# seperate the seasonal data of the players who played in this match
df_mseason = df_merged_sq.set_index('Player').reindex(df_merged_match.Player).reset_index()
df_mseason_p90 = df_merged_p90_sq.set_index('Player').reindex(df_merged_match.Player).reset_index()

for dfs in [df_merged_match, df_mseason, df_mseason_p90]:
    dfs.fillna(0, inplace=True)
# Compare these tables by minutes played:
# if the seasonstat/season_minutes < matchstat/matchmins -> overperform the avg
# -> matchstat > seasonstat*matchmins/seasonmins
# or 90mins for everybody

# and not every stat is positive, so let's make a dictionary to know the significance of performance
# edit: it might be good segmenting stats to groups so let's also do that, and rename them
combined_stats = fbref.stats_dict()

#missing_keys = []
#for c in df_merged0.columns[5:]:
#    if c in combined_stats.keys():
#        pass
#    else:
#        missing_keys.append(c)

#%% Filter statistics by correlation
corr_matrix = df_merged_p90.iloc[:, 6:].corr().abs()

# Define redundant metrics
d_redundant_stats = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.8:
            d_redundant_stats.append(corr_matrix.columns[i])

# Define non-redundant stats
d_filtered_keys = []
for k in combined_stats.keys():
    if k not in d_redundant_stats:
        d_filtered_keys.append(k)

#%% Create performance difference list
def op_list_append(listname, plus_minus, overpercent_dict):
    for x in overpercent_dict.keys():
        if plus_minus == '+':
            globals()[f'{x}'] = +overpercent_dict.get(x)
        elif plus_minus == '-':
            globals()[f'{x}'] = -overpercent_dict.get(x)
        
    listname.append({'squad': squad,
                     'position': pos,
                     'winger':winger,
                     'player': player,
                     'stat_category': stat_category,
                     'stat': stat_name_long,
                     'overperformance%_season': overpercent_season,
                     'match_value': matchstat,
                     'season_avg_value': season_avg,
                     'season_value': seasonstat,
                     'league_avg': league_avg,
                     'overperformance%_league': overpercent_league,
                     'pos_avg': pos_avg,
                     'overperformance%_pos': overpercent_pos,
                     'redundant': redundancy})

overperform_list = []
for i, row in df_merged_match.iterrows():
    squad, player, pos, winger = row[['Squad', 'Player', 'pos_group', 'winger']]
    matchmins = 90 #df_merged_match.Min[row]
    seasonmins = df_mseason.Min[i]
    
    for col in range(6, len(df_merged_match.columns[:-2])):
        stat_name = df_merged_match.columns[col]
        stat_category = combined_stats.get(stat_name).get('category')
        stat_name_long = combined_stats.get(stat_name).get('name')
        matchstat = df_merged_match.iloc[i, col]
        seasonstat = df_mseason.iloc[i, col]
        season_avg = df_mseason_p90.iloc[i, col]
        league_avg = league_avg_dict.get(stat_name)
        pos_avg = pos_avg_dict.get(pos).get(stat_name)
        redundancy = True if stat_name in d_redundant_stats else ('ERROR' if stat_name not in d_filtered_keys else False)
        
        overpercent_dict = {}
        for t in ['season', 'league', 'pos']:
            if (matchstat != 0) & (globals()[f'{t}_avg'] != 0):
                overpercent_dict[f'overpercent_{t}'] = round(((matchstat / globals()[f'{t}_avg'] -1)*100))
            elif (matchstat == 0) & (globals()[f'{t}_avg'] != 0):
                overpercent_dict[f'overpercent_{t}'] = -100
            elif (matchstat == 0) & (globals()[f'{t}_avg'] != 0):
                overpercent_dict[f'overpercent_{t}'] = 100
            elif (matchstat == 0) & (globals()[f'{t}_avg'] == 0):
                overpercent_dict[f'overpercent_{t}'] = 0
        
        
        # see if the stat is a negative or positive one
        if combined_stats.get(stat_name).get('significance') == 'positive':
            op_list_append(overperform_list, '+', overpercent_dict)
        elif combined_stats.get(stat_name).get('significance') == 'negative':
            op_list_append(overperform_list, '-', overpercent_dict)

df_overperform_merged = pd.DataFrame(overperform_list)

#%% To excel
path = r'fbref\long_short_OP.xlsx'
df_overperform_merged.to_excel(path, index=False)

#%%
from sklearn.decomposition import FactorAnalysis
# Faktoranalízis 5 főkomponenssel
df_merged_p90_forFA = df_merged_p90.copy().dropna()
fa = FactorAnalysis(n_components=5)
factors = fa.fit_transform(df_merged_p90_forFA.iloc[:, 6:])

