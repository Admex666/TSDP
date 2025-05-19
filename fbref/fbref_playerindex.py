import pandas as pd
from fbref import fbref_module as fbref
import numpy as np

df_lewa = fbref.get_all_player_data('GER', year='2020-2021')
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
'Receiving_Rec',
'Carries_Mis',
'Carries_Dis'
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
