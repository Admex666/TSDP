#%% Fetch data
import pandas as pd
from fbref import fbref_module as fbref
import numpy as np

df_lewa = fbref.get_all_player_data('GER', year='2020-2021')

#%% 
df_lewa_att = df_lewa[df_lewa.Pos.str.contains('MF') | df_lewa.Pos.str.contains('FW')].reset_index()

stats_list = [
'Performance_Gls',
'Expected_npxG',
'Standard_SoT',
'KP',
'Ast',
'Expected_xAG',
'Take-Ons_Att',
'Take-Ons_Succ',
'Carries_1/3',
'Touches_Att 3rd',
'Tackles_Att 3rd',
'Challenges_Tkl',
'Performance_Off',
'Performance_Fls',
'Performance_Fld',
'Receiving_Rec',
'Carries_Mis',
'Carries_Dis',
'Int',
'Total_PrgDist',
'Total_Cmp%',
'SCA_SCA'
]
cols_basic = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']

df_l_a_filter = df_lewa_att[cols_basic+stats_list]
df_l_a_filter = df_l_a_filter[df_l_a_filter['90s'] > 0]

# per90
stats_list90 = stats_list.copy()
df_l_a_f90 = df_l_a_filter.copy()
for i, col in enumerate(stats_list):
    if ('90' in col) or ('%' in col):
        pass
    else:
        df_l_a_f90[col] = df_l_a_f90[col]/df_l_a_f90['90s']
        df_l_a_f90.rename(columns={col:f'{col}_p90'}, inplace=True)
        stats_list90[i] = f'{col}_p90'

df90_perc = df_l_a_f90.copy()
for col90 in stats_list90:
    df90_perc[col90] = round(df90_perc[col90].rank(pct=True)*100,0)

for i in df90_perc.index:
    df90_perc.loc[i, 'perc_mean'] = df90_perc.loc[i, stats_list90].mean()
    
# Gnabry 2020/21: Goals, npxG, KeyPasses, xA, Carries 1/3, Tackles Att3rd, Shot-creating actions
cols_gnabry = ['Performance_Gls_p90', 'Expected_npxG_p90', 'KP_p90', 'SCA_SCA_p90',
               'Expected_xAG_p90', 'Carries_1/3_p90', 'Tackles_Att 3rd_p90']
weights = [1, 0.5, 1.25, 1.5, 1.25, 1, 1]

def weighted_average(values, weights):
    if len(values) == len(weights):
        sum_of_lists = 0
        for a, b in zip(values, weights):
            sum_of_lists += a*b
        return sum_of_lists / sum(weights)
    else:
        return "Error"

df90_perc_gnabry = df90_perc[cols_basic+cols_gnabry].copy()
for i, row in df90_perc_gnabry.iterrows():
    values_row = row[cols_gnabry].to_list()
    df90_perc_gnabry.loc[i, 'Gnabry_index'] = weighted_average(values_row, weights)

#%% Gnabry index on 24/25 season, bundesliga
df_2425 = fbref.get_all_player_data('GER', year='2024-2025')
filter_ = (df_2425.Pos.str.contains('MF') | df_2425.Pos.str.contains('FW')) & (df_2425['90s'] > 0)
df_2425_f = df_2425.loc[filter_,cols_basic+stats_list].copy()
# per90
df_2425_f90 = df_2425_f.copy()
for i, col in enumerate(stats_list):
    if ('90' in col) or ('%' in col):
        pass
    else:
        df_2425_f90[col] = df_2425_f90[col]/df_2425_f90['90s']
        df_2425_f90.rename(columns={col:f'{col}_p90'}, inplace=True)
        stats_list90[i] = f'{col}_p90'
# Percentile values
df90_2425_perc = df_2425_f90.copy()
for col90 in stats_list90:
    df90_2425_perc[col90] = round(df90_2425_perc[col90].rank(pct=True)*100,0)
for i in df90_2425_perc.index:
    df90_2425_perc.loc[i, 'perc_mean'] = df90_2425_perc.loc[i, stats_list90].mean()

df90_2425_perc_gnabry = df90_2425_perc[cols_basic+cols_gnabry].copy()
for i, row in df90_2425_perc_gnabry.iterrows():
    values_row = row[cols_gnabry].to_list()
    df90_2425_perc_gnabry.loc[i, 'Gnabry_index'] = weighted_average(values_row, weights)