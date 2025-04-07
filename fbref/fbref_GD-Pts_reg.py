# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:22:30 2024

@author: Adam
"""

from fbref import fbref_module as fbr
import matplotlib.pyplot as plt
import pandas as pd
URL = 'https://fbref.com/en/comps/46/NB-I-Stats'

df_table = fbr.read_html_upd(URL, 'results2024-2025461_overall')

#%% Formatting, calculating
df_table = df_table[0]
df_table = df_table.iloc[:,0:10]

[r2_gdpts, b0_gdpts, b1_gdpts] = fbr.linreg(df_table.GD, df_table.Pts)
df_table['reg_Pts'] = b0_gdpts + df_table['GD'] * b1_gdpts
df_table['reg_Pts_diff'] = df_table.Pts - df_table.reg_Pts

if 'xGD' in df_table.columns:
    [r2_xgdpts, b0_xgdpts, b1_xgdpts] = fbr.linreg(df_table.xGD, df_table.Pts)
    df_table['reg_xPts'] = b0_xgdpts + df_table['xGD'] * b1_xgdpts 
    df_table['reg_xPts_diff'] = df_table.Pts - df_table.reg_xPts

#%% Exporting to excel
output_path = 'fbref/GD-Pts_reg.xlsx'
df_table.to_excel(output_path, index=False)
