# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:04:02 2024

@author: Adam
"""

# Importing libraries
import fbref_module as fbref
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

# choose from ENG, ESP, GER, ITA...
league = 'UEL'
pos = 'FW' # FW, MF, DF, GK
matches_at_least = 2
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

[df_kpi_player_fact, df_kpi_player_dim] = create_df_kpi(df_analyse, 22)
player_name = df_kpi_player_dim[df_kpi_player_dim.Statistic=='Player']['Value'].get(1)

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

#%% KPI comparison prepare for export
player_index_year = df_analyse_year[df_analyse_year.Player == player_name].index[0]
[df_kpi_player_fact_year, df_kpi_player_dim_year] = create_df_kpi(df_analyse_year, player_index_year)

path = r'C:\TSDP\fbref\scout_summary.xlsx'
if df_kpi_player_dim.loc[1,'Value'] == df_kpi_player_dim_year.loc[1,'Value']:
    df_kpi_player_fact_both = pd.merge(df_kpi_player_fact,
                                       df_kpi_player_fact_year,
                                       on='Statistic',
                                       how='inner')
    df_kpi_player_fact_both.columns = ['Statistic', 'Value', 'Percentile',
                                       'Value_year', 'Percentile_year']
    df_kpi_player_fact_both.to_excel(path, index=False)
    print("Names were matching.")
else:
    print("Error: name is not matching")
    