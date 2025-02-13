# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:24:05 2025

@author: Adam
"""
# Set and scrape league dataframes
import pandas as pd
import fbref_module as fbref
from bs4 import BeautifulSoup
import requests

league = 'FRA'
URL_match = 'https://fbref.com/en/matches/7f0d9a30/Brest-Paris-Saint-Germain-February-11-2025-Champions-League'

comp_id, league_name = fbref.team_dict_get(league)

#URLs
URL_standard = (
    "https://fbref.com/en/comps/" 
    + comp_id
    + '/stats/'
    + league_name
    + '-Stats#all_stats_'
    + 'standard'
)

stats_list = ['passing', 'possession', 'shooting', 'defense', 'misc']
for stat in stats_list:
    globals()[f'URL_{stat}'] = (
        "https://fbref.com/en/comps/" 
        + comp_id
        + f'/{stat}/'
        + league_name
        + '-Stats#all_stats_'
        + stat
    )

del comp_id, league, stat

#%% Scrape the competition stat dfs
stats_list.append('standard')
for stat in stats_list:
    globals()[f'df_{stat}'] = fbref.format_column_names(fbref.scrape(globals()[f'URL_{stat}'], f'stats_{stat}'))

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

#%% Create dataframes that include all stats
# Only want relevant squads' season data
if (teams_df.team_name[0] in df_standard.Squad.unique()) & (teams_df.team_name[1] in df_standard.Squad.unique()):
    for stat in stats_list:
        globals()[f'df_{stat}'] = globals()[f'df_{stat}'][globals()[f'df_{stat}'].Squad.isin([teams_df.team_name[0], teams_df.team_name[1]])]
    print('Team names found')
else:
    print('Error: team names not found')

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
                         how='inner', on='Player', 
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

#%% Find and rename the columns in merged season df
df_merged.rename(columns={'Playing Time_Min': 'Min',
                          'Standard_Sh':'Performance_Sh',
                          'Standard_SoT': 'Performance_SoT',
                          },
                 inplace=True)
# Remove unnecessary columns for match dfs
for team in [0,1]:
    globals()[f'df_merged{team}'].drop(columns=missing_cols[4:]+['#'],
                                       inplace=True)
    # and remove unnecessary summarizing row
    globals()[f'df_merged{team}'] = globals()[f'df_merged{team}'].iloc[:-1,:]
    
#%% Comparing stats
df_merged = df_merged.loc[:,df_merged0.columns.unique()]
df_merged.iloc[:,5:] = df_merged.iloc[:,5:].astype(float)
# seperate the seasonal data of the players who played in this match
df_mseason0 = df_merged.set_index('Player').reindex(df_merged0.Player).reset_index()
df_mseason1 = df_merged.set_index('Player').reindex(df_merged1.Player).reset_index()

for dfs in [df_merged0, df_merged1, df_mseason0, df_mseason1]:
    dfs.fillna(0, inplace=True)
# Compare these tables by minutes played:
# if the seasonstat/season_minutes < matchstat/matchmins -> overperform the avg
# -> matchstat > seasonstat*matchmins/seasonmins
# or 90 for everybody, don't care of played mins

# and not every stat is positive, so let's make a dictionary to know the significance of performance
# edit: it might be good segmenting stats to groups so let's also do that, and rename them
combined_stats = {
    'Performance_Sh': {'name': 'Shots', 'category': 'Offensive', 'significance': 'positive'},
    'Carries_Carries': {'name': 'Carries', 'category': 'Possession', 'significance': 'positive'},
    'Carries_PrgC': {'name': 'Progressive Carries', 'category': 'Possession', 'significance': 'positive'},
    'Total_Cmp': {'name': 'Passes Completed', 'category': 'Passing', 'significance': 'positive'},
    'Total_Cmp%': {'name': 'Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
    'Total_TotDist': {'name': 'Total Passing Distance', 'category': 'Passing', 'significance': 'positive'},
    'Total_PrgDist': {'name': 'Progressive Passing Distance', 'category': 'Passing', 'significance': 'positive'},
    'Short_Cmp%': {'name': 'Short Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
    'Medium_Cmp': {'name': 'Medium Passes Completed', 'category': 'Passing', 'significance': 'positive'},
    'Medium_Att': {'name': 'Medium Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
    'Medium_Cmp%': {'name': 'Medium Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
    'Long_Cmp': {'name': 'Long Passes Completed', 'category': 'Passing', 'significance': 'positive'},
    'Long_Cmp%': {'name': 'Long Pass Completion %', 'category': 'Passing', 'significance': 'positive'},
    '1/3': {'name': 'Passes into Final Third', 'category': 'Passing', 'significance': 'positive'},
    'PPA': {'name': 'Passes into Penalty Area', 'category': 'Passing', 'significance': 'positive'},
    'PrgP': {'name': 'Progressive Passes', 'category': 'Passing', 'significance': 'positive'},
    'Tackles_Tkl': {'name': 'Tackles', 'category': 'Defensive', 'significance': 'positive'},
    'Tackles_Att 3rd': {'name': 'Tackles in Attacking Third', 'category': 'Defensive', 'significance': 'positive'},
    'Challenges_Tkl': {'name': 'Dribblers Tackled', 'category': 'Defensive', 'significance': 'positive'},
    'Challenges_Att': {'name': 'Dribbles Challenged', 'category': 'Defensive', 'significance': 'positive'},
    'Challenges_Tkl%': {'name': 'Tackle Success %', 'category': 'Defensive', 'significance': 'positive'},
    'Tkl+Int': {'name': 'Tackles + Interceptions', 'category': 'Defensive', 'significance': 'positive'},
    'Touches_Att 3rd': {'name': 'Touches in Attacking Third', 'category': 'Possession', 'significance': 'positive'},
    'Carries_TotDist': {'name': 'Total Carrying Distance', 'category': 'Possession', 'significance': 'positive'},
    'Carries_PrgDist': {'name': 'Progressive Carrying Distance', 'category': 'Possession', 'significance': 'positive'},
    'Carries_1/3': {'name': 'Carries into Final Third', 'category': 'Possession', 'significance': 'positive'},
    'Carries_Mis': {'name': 'Miscontrols', 'category': 'Possession', 'significance': 'negative'},
    'Receiving_Rec': {'name': 'Passes Received', 'category': 'Possession', 'significance': 'positive'},
    'Receiving_PrgR': {'name': 'Progressive Passes Received', 'category': 'Possession', 'significance': 'positive'},
    'Performance_Fls': {'name': 'Fouls Committed', 'category': 'Miscellaneous', 'significance': 'negative'},
    'Performance_Recov': {'name': 'Ball Recoveries', 'category': 'Defensive', 'significance': 'positive'},
    'Aerial Duels_Won': {'name': 'Aerial Duels Won', 'category': 'Aerial Duels', 'significance': 'positive'},
    'Aerial Duels_Lost': {'name': 'Aerial Duels Lost', 'category': 'Aerial Duels', 'significance': 'negative'},
    'Aerial Duels_Won%': {'name': 'Aerial Duels Won %', 'category': 'Aerial Duels', 'significance': 'positive'},
    'Performance_SoT': {'name': 'Shots on Target', 'category': 'Offensive', 'significance': 'positive'},
    'Expected_xG': {'name': 'Expected Goals', 'category': 'Offensive', 'significance': 'positive'},
    'Expected_npxG': {'name': 'Non-Penalty Expected Goals', 'category': 'Offensive', 'significance': 'positive'},
    'Long_Att': {'name': 'Long Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
    'KP': {'name': 'Key Passes', 'category': 'Offensive', 'significance': 'positive'},
    'CrsPA': {'name': 'Crosses into Penalty Area', 'category': 'Passing', 'significance': 'positive'},
    'Touches_Att Pen': {'name': 'Touches in Attacking Penalty Area', 'category': 'Possession', 'significance': 'positive'},
    'Take-Ons_Tkld': {'name': 'Take-Ons Tackled', 'category': 'Possession', 'significance': 'negative'},
    'Take-Ons_Tkld%': {'name': 'Take-Ons Tackled %', 'category': 'Possession', 'significance': 'negative'},
    'Performance_Crs': {'name': 'Crosses', 'category': 'Passing', 'significance': 'positive'},
    'Tackles_Mid 3rd': {'name': 'Tackles in Middle Third', 'category': 'Defensive', 'significance': 'positive'},
    'Take-Ons_Succ%': {'name': 'Successful Take-Ons %', 'category': 'Possession', 'significance': 'positive'},
    'Carries_Dis': {'name': 'Dispossessed', 'category': 'Possession', 'significance': 'negative'},
    'Take-Ons_Att': {'name': 'Take-Ons Attempted', 'category': 'Possession', 'significance': 'positive'},
    'Take-Ons_Succ': {'name': 'Successful Take-Ons', 'category': 'Possession', 'significance': 'positive'},
    'Total_Att': {'name': 'Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
    'Tackles_TklW': {'name': 'Tackles Won', 'category': 'Defensive', 'significance': 'positive'},
    'Challenges_Lost': {'name': 'Challenges Lost', 'category': 'Defensive', 'significance': 'negative'},
    'Blocks_Blocks': {'name': 'Blocks', 'category': 'Defensive', 'significance': 'positive'},
    'Blocks_Pass': {'name': 'Passes Blocked', 'category': 'Defensive', 'significance': 'positive'},
    'Clr': {'name': 'Clearances', 'category': 'Defensive', 'significance': 'positive'},
    'Touches_Touches': {'name': 'Touches', 'category': 'Possession', 'significance': 'positive'},
    'Touches_Live': {'name': 'Live-Ball Touches', 'category': 'Possession', 'significance': 'positive'},
    'Carries_CPA': {'name': 'Carries into Penalty Area', 'category': 'Possession', 'significance': 'positive'},
    'Performance_Off': {'name': 'Offsides', 'category': 'Miscellaneous', 'significance': 'negative'},
    'Performance_TklW': {'name': 'Tackles Won', 'category': 'Defensive', 'significance': 'positive'},
    'Performance_Ast': {'name': 'Assists', 'category': 'Offensive', 'significance': 'positive'},
    'Ast': {'name': 'Assists', 'category': 'Offensive', 'significance': 'positive'},
    'Expected_xAG': {'name': 'Expected Assisted Goals', 'category': 'Offensive', 'significance': 'positive'},
    'xAG': {'name': 'Expected Assists', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_Fld': {'name': 'Fouls Drawn', 'category': 'Miscellaneous', 'significance': 'positive'},
    'Performance_Gls': {'name': 'Goals', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_Int': {'name': 'Interceptions', 'category': 'Defensive', 'significance': 'positive'},
    'Int': {'name': 'Interceptions', 'category': 'Defensive', 'significance': 'positive'},
    'Performance_Sh/90': {'name': 'Shots per 90', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_PK': {'name': 'Penalties Scored', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_PKatt': {'name': 'Penalties Attempted', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_CrdY': {'name': 'Yellow Cards', 'category': 'Disciplinary', 'significance': 'negative'},
    'Performance_CrdR': {'name': 'Red Cards', 'category': 'Disciplinary', 'significance': 'negative'},
    'Short_Cmp': {'name': 'Short Passes Completed', 'category': 'Passing', 'significance': 'positive'},
    'Short_Att': {'name': 'Short Passes Attempted', 'category': 'Passing', 'significance': 'positive'},
    'Tackles_Def 3rd': {'name': 'Tackles in Defensive Third', 'category': 'Defensive', 'significance': 'positive'},
    'Blocks_Sh': {'name': 'Shots Blocked', 'category': 'Defensive', 'significance': 'positive'},
    'Err': {'name': 'Errors Leading to Shots', 'category': 'Defensive', 'significance': 'negative'},
    'Touches_Def Pen': {'name': 'Touches in Defensive Penalty Area', 'category': 'Defensive', 'significance': 'positive'},
    'Touches_Def 3rd': {'name': 'Touches in Defensive Third', 'category': 'Defensive', 'significance': 'positive'},
    'Touches_Mid 3rd': {'name': 'Touches in Middle Third', 'category': 'Defensive', 'significance': 'positive'},
    'Performance_2CrdY': {'name': 'Second Yellow Cards', 'category': 'Disciplinary', 'significance': 'negative'},
    'Performance_PKwon': {'name': 'Penalties Won', 'category': 'Offensive', 'significance': 'positive'},
    'Performance_PKcon': {'name': 'Penalties Conceded', 'category': 'Defensive', 'significance': 'negative'},
    'Performance_OG': {'name': 'Own Goals', 'category': 'Defensive', 'significance': 'negative'}
}
#missing_keys = []
#for c in df_merged0.columns[5:]:
#    if c in combined_stats.keys():
#        pass
#    else:
#        missing_keys.append(c)

def op_list_append(listname, plus_minus, overpercent):
    if plus_minus == '+':
        overpercent = +overpercent
    elif plus_minus == '-':
        overpercent = -overpercent
        
    listname.append({'squad': squad,
                     'player': player,
                     'stat_category': stat_category,
                     'stat': stat_name_long,
                     'overperformance%': overpercent,
                     'match_avg_value': match_avg,
                     'season_avg_value': season_avg,
                     'match_value': matchstat,
                     'season_value': seasonstat})

for team in [0,1]:
    overperform_list = []
    for row in range(len(globals()[f'df_merged{team}'])):
        squad = globals()[f'df_merged{team}'].Squad[row]
        player = globals()[f'df_merged{team}'].Player[row]
        matchmins = 90 #globals()[f'df_merged{team}'].Min[row]
        seasonmins = globals()[f'df_mseason{team}'].Min[row]
        
        for col in range(6, len(globals()[f'df_merged{team}'].columns)):
            stat_name = globals()[f'df_merged{team}'].columns[col]
            stat_category = combined_stats.get(stat_name).get('category')
            stat_name_long = combined_stats.get(stat_name).get('name')
            matchstat = globals()[f'df_merged{team}'].iloc[row, col]
            seasonstat = globals()[f'df_mseason{team}'].iloc[row, col]
            
            if '%' in stat_name:
                match_avg = matchstat
                season_avg = seasonstat
                if matchstat == 0:
                    if matchstat == seasonstat:
                        overpercent = 0
                    else:
                        overpercent = -100
                elif seasonstat == 0:
                    overpercent = 100
                else:
                    overpercent = round((match_avg/season_avg-1)*100)
                    # see if the stat is a negative or positive one
                if combined_stats.get(stat_name).get('significance') == 'positive':
                    op_list_append(overperform_list, '+', overpercent)
                elif combined_stats.get(stat_name).get('significance') == 'negative':
                    op_list_append(overperform_list, '-', overpercent)

            else:
                match_avg = matchstat
                season_avg = seasonstat/seasonmins*90
                if matchstat == 0:
                    if matchstat == seasonstat:
                        overpercent = 0 
                    else:
                        overpercent = -100
                elif seasonstat == 0:
                    overpercent = 100
                else:
                    overpercent = round(((matchstat / (seasonstat*matchmins/seasonmins)-1)*100))
                    # see if the stat is a negative or positive one
                if combined_stats.get(stat_name).get('significance') == 'positive':
                    op_list_append(overperform_list, '+', overpercent)
                elif combined_stats.get(stat_name).get('significance') == 'negative':
                    op_list_append(overperform_list, '-', overpercent)

            
    globals()[f'df_overperform{team}'] = pd.DataFrame(overperform_list)

df_overperform_merged = pd.concat([df_overperform0, df_overperform1])    

#%% To excel
path = r'C:\Users\Ádám\Dropbox\TSDP\fbref\long_short_OP.xlsx'
df_overperform_merged.to_excel(path, index=False)
