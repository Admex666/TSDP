# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:04:02 2024

@author: Adam
"""

# Importing libraries
from fbref import fbref_module as fbref
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

# choose from ENG, ESP, GER, ITA...
league = 'GER'
pos = 'DF' # FW, MF, DF, GK
matches_at_least = 4
year = '2023-2024' # the year of comparison

#%%
def year_of_url(URL, year):
    URL_list1 = URL.split('/')[:6]
    URL_list2 = URL.split('/')[6]
    URL_list3 = year + '-' + URL.split('/')[7]
    URL_new_list = URL_list1
    URL_new_list.append(year)
    URL_new_list.append(URL_list2)
    URL_new_list.append(URL_list3)
    URL_new = '/'.join(URL_new_list)
    return URL_new

comp_id, league_name = fbref.team_dict_get(league)

#%% 
if pos == 'FW':
    # FORWARD KPIs: MinPlayed(/90), G-PK, npxG, assists, xA(G), shot-creating act./key_pass,
    # Pass_cmp%, prog_pass, prog_carr, 
    # Tackles, interceptions, aerials won
    stats_list = ['passing','defense', 'gca', 'standard']
    url_list = ['passing','defense', 'gca', 'stats']
    
    cols_standard = ['Per 90 Minutes_G-PK', 'Per 90 Minutes_npxG', 
                     'Per 90 Minutes_Ast', 'Per 90 Minutes_xAG', 
                     'Progression_PrgP', 'Progression_PrgC','Progression_PrgR']
    cols_passing = ['KP', 'Total_Cmp%']
    cols_defense = ['Tackles_Tkl', 'Int']
    cols_gca = ['SCA_SCA90']
    
    cols_all_list = cols_standard + cols_passing + cols_defense + cols_gca

elif pos == 'MF':
    # MIDFIELD KPIs: MinPlayed(/90), KeyPass90, xAG90, ChancesCreated90, (DribblesMade%),
    # ProgPass90, ProgCarry90, Tackle%, Int90
    stats_list = ['passing','defense', 'gca', 'standard']
    url_list = ['passing','defense', 'gca', 'stats']
    
    cols_standard = ['Per 90 Minutes_xAG', 'Progression_PrgP', 'Progression_PrgC']
    cols_passing = ['KP']
    cols_defense = ['Challenges_Tkl%', 'Int']
    cols_gca = ['SCA_SCA90']
    cols_all_list = cols_standard + cols_passing + cols_defense + cols_gca
    
elif pos == 'DF':
    # (central) DEFENDER KPIs: Headers(%), Tackle(%), 
    # Int, Pass%, Error, Block, Clearance
    stats_list = ['passing', 'defense', 'misc']
    url_list = ['passing', 'defense', 'misc']

    cols_passing = ['Total_Cmp%', 'Total_Att']
    cols_defense = ['Challenges_Tkl%', 'Challenges_Att', 
                    'Err', 'Int', 'Blocks_Blocks', 'Clr']
    cols_misc = ['Aerial Duels_Won%', 'Aerial Duels_Won']
    cols_all_list = cols_passing + cols_defense + cols_misc

elif pos == 'GK':
    # Goalkeeper KPIs: Save%, xG+/-, Sh/GoalsAG, Pass%, Err, PrgPass, Launched
    stats_list = ['keeper', 'keeper_adv', 'defense', 'passing']
    url_list = ['keepers', 'keepersadv', 'defense', 'passing']

    cols_keeper = ['Performance_Save%', 'Performance_SoTA', 'Performance_GA']
    cols_keeper_adv = ['Expected_PSxG+/-', 'Passes_Launch%']   
    cols_defense = ['Err']
    cols_passing = ['Total_Cmp%']
    cols_all_list = cols_keeper + cols_keeper_adv + cols_defense + cols_passing
    
else:
    pass

for stat, url in zip(stats_list, url_list):
    globals()[f'URL_{stat}'] = ("https://fbref.com/en/comps/" + comp_id
                                + f'/{url}/' + league_name 
                                + '-Stats#all_stats_' + stat)
    globals()[f'URL_{stat}_year'] = ("https://fbref.com/en/comps/" + comp_id 
                                     + f'/{year}/{url}/{year}-' 
                                     + league_name + '-Stats')

#%% Scraping the data
for stat in stats_list:
    for y in ['', '_year']: 
        globals()[f'df_{stat}{y}'] = fbref.format_column_names(fbref.scrape(globals()[f'URL_{stat}{y}'], f'stats_{stat}'))
        # Cleaning header rows
        globals()[f'df_{stat}{y}'].drop(globals()[f'df_{stat}{y}'][globals()[f'df_{stat}{y}']['Rk']=='Rk'].index, inplace=True)
        
#%% Function for creating table for analysis
def create_df_analyse(pos, matches_at_least, stats_list, cols_all_list, year):
    cols_to_include = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
    if year == True:
        y = '_year'
    elif year == False:
        y = ''
    
    df_analyse = globals()[f'df_{stats_list[0]}{y}'].copy()
    
    if 'Playing Time_90s' in df_analyse.columns:
        df_analyse.rename(columns={'Playing Time_90s': '90s'}, inplace=True)
    df_analyse.dropna(inplace=True)
    
    # Filter by position
    df_analyse = df_analyse.loc[df_analyse.Pos.str.contains(pos),:]
    # Filter by playing time
    df_analyse['90s'] = df_analyse['90s'].astype(float)
    df_analyse = df_analyse[df_analyse['90s'] >= matches_at_least].reset_index(drop=True)

    # Building tables for analysis
    for stat in stats_list[1:]:
        df_analyse = pd.merge(df_analyse, globals()[f'df_{stat}{y}'],
                              on=['Player', 'Squad'], how='left',
                              suffixes=['','_remove'])
        # Remove the duplicate columns
        df_analyse.drop([i for i in df_analyse.columns if 'remove' in i],
                       axis=1, inplace=True)
    
    df_analyse = pd.concat([df_analyse.loc[:, cols_to_include],
                            df_analyse.loc[:, cols_all_list].astype(float)],
                           axis=1)

    # Make every column per90
    for col in cols_all_list:
        if ('90' in col) or ('%' in col):
            pass
        else:
            df_analyse.loc[:,col] = df_analyse.loc[:,col]/df_analyse['90s']
            col_new = col + '_p90'
            df_analyse.rename(columns={col:col_new}, inplace=True)
    
    if pos == 'GK':
        df_analyse['SoTA/GA'] = df_analyse.Performance_SoTA_p90 / df_analyse.Performance_GA_p90
        df_analyse.drop(columns=['Performance_SoTA_p90', 'Performance_GA_p90'], inplace=True)
    
    # Convert Age and Born to integer
    df_analyse.Age = df_analyse.Age.str.split('-').str.get(0).astype(int)
    df_analyse.Born = df_analyse.Born.astype(int)
    
    df_analyse.fillna(0, inplace=True)
    df_analyse = df_analyse.reset_index(drop=True)   
    return df_analyse

#%% Create analyse table
df_analyse = create_df_analyse(pos, matches_at_least, stats_list, cols_all_list, year=False)
df_analyse_year = create_df_analyse(pos, matches_at_least, stats_list, cols_all_list, year=True)

cols_to_include = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']
cols_all_list_p90 = df_analyse.columns.to_list()
for c in cols_to_include:
    if c in cols_all_list_p90:
        cols_all_list_p90.remove(c)
        
#%% Creating a table of percentiles
def create_df_percentiles(analysis_df):
    df_percentiles = analysis_df.copy()
    for col in cols_all_list_p90:
        for row in range(len(df_percentiles[col])):
            value = analysis_df.loc[row,col]
            percentile = round(percentileofscore(analysis_df[col], value, kind='rank'), 0)
            df_percentiles.loc[row, col] = percentile
    
    df_percentiles['Combined'] = None
    for row in range(len(df_percentiles)):
        df_percentiles.loc[row, 'Combined'] = df_percentiles.loc[row, cols_all_list_p90].sum()

    return df_percentiles

df_percentiles = create_df_percentiles(df_analyse)
df_percentiles_year = create_df_percentiles(df_analyse_year)

#%% KPI Percentile improvement from last season
df_percentiles_diff = pd.merge(df_percentiles[cols_to_include],
                               df_percentiles_year['Player'],
                               how='inner', on='Player')

for col in cols_all_list_p90:
    df_percentiles_diff[col] = df_percentiles[col] - df_percentiles_year[col]
del col

df_percentiles_diff['diff_sum'] = [row.loc[cols_all_list_p90].sum() for i, row in df_percentiles_diff.iterrows()]

#%% Final table of KPIs for radar chart: Stat, Value, Percentile
def create_df_kpi(analysis_df, player_index):
    df_kpi_player = analysis_df.T[player_index].reset_index()
    df_kpi_player.columns = ['Statistic', 'Value']
    # split the string and numeric values
    df_kpi_player_dim = df_kpi_player.loc[0:7,:]
    df_kpi_player.drop(index=(range(0,8)), inplace=True)
    df_kpi_player = df_kpi_player.reset_index(drop=True)
    
    df_kpi_player['Percentile'] = None
    for metric in df_kpi_player['Statistic'].unique():
        v = df_kpi_player.loc[df_kpi_player['Statistic'] == metric,'Value']
        value = v.iloc[0]
        percentile = round(percentileofscore(analysis_df[metric], value, kind='rank'), 0)
        df_kpi_player.loc[df_kpi_player.Statistic == metric, 'Percentile'] = percentile
    
    return [df_kpi_player, df_kpi_player_dim]

# Choose the player
player_index = 98
[df_kpi_player_fact, df_kpi_player_dim] = create_df_kpi(df_analyse, player_index)
player_name = df_kpi_player_dim[df_kpi_player_dim.Statistic=='Player']['Value'].get(1)

player_index_year = df_analyse_year[df_analyse_year.Player == player_name]
if player_index_year.empty:
    [df_kpi_player_fact_year, df_kpi_player_dim_year] = [df_kpi_player_fact.copy(), df_kpi_player_dim.copy()]
    df_kpi_player_fact_year[['Value', 'Percentile']] = 0
    df_kpi_player_dim_year[['Value', 'Percentile']] = None
    print(f'Player index not found in {year} dataframe.')
else:
    player_index_year = player_index_year.index[0]
    [df_kpi_player_fact_year, df_kpi_player_dim_year] = create_df_kpi(df_analyse_year, player_index_year)
    print(f'Index of {player_name} found in {year} dataframe.')

df_kpi_player_fact_year.columns = ['Statistic_year', 'Value_year', 'Percentile_year']

#%% KPI comparison prepare for export
df_kpi_player_fact_both = pd.concat([df_kpi_player_fact, df_kpi_player_fact_year], axis=1)

if df_kpi_player_dim.loc[1,'Value'] == df_kpi_player_dim_year.loc[1,'Value']:
    print("Names were matching.")
else:
    print("Names are not matching")
    
stats_dict = fbref.stats_dict()
if df_kpi_player_fact_both.Statistic.all() == df_kpi_player_fact_both.Statistic_year.all():
    df_kpi_player_fact_both.drop(columns='Statistic_year', inplace=True)
    df_kpi_player_fact_both['Statistic_pretty'] = None
    
    for i, stat in enumerate(df_kpi_player_fact_both.Statistic):
        stat_pretty = stat.replace('_p90', '')
        if stat_pretty in stats_dict.keys():
            if '_p90' in stat:
                df_kpi_player_fact_both.loc[i, 'Statistic_pretty'] = stats_dict[stat_pretty]['name']+' per 90'
            else:
                df_kpi_player_fact_both.loc[i, 'Statistic_pretty'] = stats_dict[stat_pretty]['name']
        else:
            df_kpi_player_fact_both.loc[i, 'Statistic_pretty'] = f'ERROR_{stat_pretty}'
    
else:
    print('Statistics are not matching.')
    
#%% To Excel
path = 'fbref\scout_summary.xlsx'
df_kpi_player_fact_both.to_excel(path, index=False)
    
#%% Visualize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Calculate the differences
differences = df_kpi_player_fact_both['Percentile'] - df_kpi_player_fact_both['Percentile_year']

# Normalize the differences for colormap
norm = mcolors.Normalize(vmin=-75, vmax=75)
cmap = plt.get_cmap('RdYlGn')  # Red-Yellow-Green colormap

# Create the plot
plt.figure(figsize=(10, 6))
y = range(len(df_kpi_player_fact_both))

# Plot the two percentiles
dot_size = 125
plt.scatter(df_kpi_player_fact_both['Percentile'], y, color='blue', label='Current season', zorder=2, s=dot_size)
plt.scatter(df_kpi_player_fact_both['Percentile_year'], y, color='red', label='Last season', zorder=2, s=dot_size)

# Add connecting lines with color based on differences
for i, (pct, pct_year, diff) in enumerate(zip(df_kpi_player_fact_both['Percentile'], df_kpi_player_fact_both['Percentile_year'], differences)):
    color = cmap(norm(diff))  # Get color from colormap based on normalized difference
    plt.plot([pct_year, pct], [i, i], color=color, linestyle='-', alpha=0.8, zorder=1)

# Prettify statistic names
stats_nop90 = df_kpi_player_fact_both['Statistic'].str.strip('_p90')
df_kpi_player_fact_both['Statistic_pretty'] = None
for i, s in enumerate(stats_nop90):
    if s in fbref.stats_dict():
        df_kpi_player_fact_both['Statistic_pretty'][i] = fbref.stats_dict().get(s).get('name')
    else:
        df_kpi_player_fact_both['Statistic_pretty'][i] = df_kpi_player_fact_both['Statistic'][i]


# Customize axes and labels
plt.yticks(y, df_kpi_player_fact_both['Statistic_pretty'])
plt.gca().invert_yaxis()  # Show first row at the top
plt.xlabel('Percentile')

# Add title and subtitle
plt.title(f'{player_name} percentile changes', fontsize=20, pad=20)
plt.suptitle('Comparison of current season vs. last season percentiles', fontsize=14, y=0.95, x=0.5, style='italic')

# Add legend
plt.legend(loc='lower center')

# Add grid for readability
plt.grid(True, axis='x', linestyle='--', alpha=0.6, zorder=1)

# Add a colorbar to show the mapping of differences to colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Percentile difference')

plt.tight_layout()
plt.show()
