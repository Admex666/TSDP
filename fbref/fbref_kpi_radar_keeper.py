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

URL = 'https://fbref.com/en/comps/11/keepers/Serie-A-Stats#all_stats_keeper'
URLadv = 'https://fbref.com/en/comps/11/keepersadv/Serie-A-Stats'

df_basic = fbref.read_html_upd(URL, 'stats_keeper')
df_adv = fbref.read_html_upd(URLadv, 'stats_keeper_adv')

#%% Cleaning, formatting
df_basic = fbref.format_column_names(fbref.column_joiner(fbref.to_dataframe(df_basic)))
df_adv = fbref.format_column_names(fbref.column_joiner(fbref.to_dataframe(df_adv)))

df_basic.drop(index=25, inplace=True)
df_adv.drop(index=25, inplace=True)

#%% Building tables for analysis
# Needed KPIs: MP(/90), PSxG+/-, GA, Save%, CS%, 
# Launch%, AvgLen, Crosses_Stp%, Sweeper_AvgDist
df_basic.columns
df_adv.columns

df_analyse = pd.merge(df_basic[['Player', 'Squad', 'Playing Time_MP',
                                'Performance_Save%', 'Performance_CS%']],
                      df_adv[['Player', 'Expected_PSxG+/-', 'Goals_GA',
                              'Goal Kicks_Launch%', 'Goal Kicks_AvgLen',
                              'Crosses_Stp%', 'Sweeper_AvgDist']],
                      on='Player')

df_analyse.info()
df_analyse.iloc[:,2:11] = df_analyse.iloc[:,2:11].astype(float)

df_analyse_per90 = df_analyse.copy()
for x in [5, 6]:
    for y in range(len(df_analyse_per90)):
        df_analyse_per90.iloc[y,x] = df_analyse_per90.iloc[y,x] / df_analyse_per90.iloc[y,2]
del x, y
#%% Final table of KPIs for radar chart: Stat, Value, Percentile
df_kpi = df_analyse_per90.T
df_kpi_player = df_kpi[29]
df_kpi_player = df_kpi_player.reset_index()

df_kpi_player.columns = ['Statistic', 'Value']

df_kpi_player_dim = df_kpi_player.iloc[0:2,:]
df_kpi_player.drop(index=[0,1], inplace=True)
df_kpi_player.Value = df_kpi_player.Value.astype(float)
df_kpi_player = df_kpi_player.reset_index(drop=True)

df_kpi_player['Percentile'] = None
for x in range(len(df_kpi_player['Statistic'].unique())):
    column_name = df_kpi_player['Statistic'].unique()[x]
    asd = df_kpi_player.loc[df_kpi_player['Statistic'] == column_name,'Value']
    value = asd.iloc[0]
    percentile = round(percentileofscore(df_analyse_per90[column_name], value, kind='rank'), 0)
    df_kpi_player['Percentile'][x] = percentile
    
#%%
df_kpi_player.to_excel(r'C:\TwitterSportsDataProject\fbref scrapes\fbref_scout_summary.xlsx', index=False)
