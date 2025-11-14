# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:12:16 2024

@author: Adam
"""

"""
Stats we need:
    1. Pass number - Pass comp% (passing)
    2. Pass number AGAINST - passAG comp% (passing dynamics)
    3. Shots number - shot% (attacking efficiency)
    4. Headers won% - headers attempted nr. (Aerial)
    5. ((Possession lost nr. - Possession won nr.)) passcmp% - passfail% (Posession)
    6. Tackles won% - tackles nr. (Tackling)

Sources (fbref dfs):
    0. standard stats
    1. passing
    2. passing against
    3. shooting
    4. misc: aerial duels
    5. passing
    6. defensive actions
    +1. (tkl+int - tkl+intAG) defensive actions
    -->
    standard, pass, passAG, shots, misc, def
    
    https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard
    https://fbref.com/en/comps/9/passing/Premier-League-Stats#all_stats_passing
    https://fbref.com/en/comps/9/shooting/Premier-League-Stats#all_stats_shooting
    https://fbref.com/en/comps/9/misc/Premier-League-Stats#all_stats_misc
"""
#%% Scrape the data from fbref
import pandas as pd
import numpy as np
from modules import fbref_module as fbref
import matplotlib.pyplot as plt

# Set parameters
league = 'FRA'
min_90_played = 5 # how many matches at least
only_position = 'MF' # DF, MF, FW or GK

df = fbref.get_all_player_data(league, year='2024-2025')

#%% Format the merged data a bit
# Filter by parameters
df_super = df.loc[df['90s'].astype(float) >= min_90_played, :]
df_super = df_super.loc[df_super['Pos'].str.contains(only_position), :]

df_super.Age = df_super.Age.str.split('-').str.get(0).astype(float) # get age as number
df_super = df_super[df_super['90s'] != 0]

for col in df_super.columns[8:]:
    if ('90' in col) or ('%' in col) or ('Playing Time' in col):
        pass
    else:
        df_super[col] = df_super[col]/df_super['90s']
        df_super.rename(columns={col:f'{col}_p90'}, inplace=True)
df_super['league'] = fbref.team_dict_get(league)[1].replace('-', ' ')

#%% To excel
path = 'player_scatters.xlsx'
df_super.to_excel(path, index=False)

#%% Plotting (defining function)

def data_to_scatter(xname, yname, xlabel, ylabel, title, ids=None):
    # Ellenőrizzük, hogy a megadott oszlopok léteznek
    if xname not in df_super.columns or yname not in df_super.columns:
        print(f"Hiba: {xname} vagy {yname} nem található az adatkeretben!")
        return
    
    # Szűrjük ki a hiányzó értékeket
    valid_data = df_super[[xname, yname, 'Player', 'Squad']].dropna()
    xcol = valid_data[xname]
    ycol = valid_data[yname]
    
    # Colors and styling
    background_color = '#3c3d3d'
    mycolor = '#5ECB43'
    highlight_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    
    fig = plt.figure(figsize=(12, 8), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    
    # Minden játékos szürke színnel
    plt.scatter(
        x=xcol,
        y=ycol,
        color='#AAAAAA',
        alpha=0.6,
        s=50,
        edgecolors='#888888',
        linewidth=0.5
    )
    
    # Kiemelt játékosok (ha meg vannak adva ID-k)
    if ids is not None:
        if isinstance(ids, list):
            highlighted_players = valid_data.loc[ids]
        else:
            highlighted_players = valid_data.loc[[ids]]
        
        for idx, (player_idx, row) in enumerate(highlighted_players.iterrows()):
            color_idx = idx % len(highlight_colors)
            plt.scatter(
                x=row[xname],
                y=row[yname],
                color=highlight_colors[color_idx],
                s=120,
                edgecolors='white',
                linewidth=2,
                alpha=0.9,
                zorder=5
            )
            
            # Szöveges címke hozzáadása
            plt.annotate(
                row['Player'],
                xy=(row[xname], row[yname]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold',
                color=highlight_colors[color_idx],
                alpha=0.9,
                ha='left',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=background_color, 
                         edgecolor=highlight_colors[color_idx], alpha=0.8)
            )
    
    # Medián vonalak
    xmedian = xcol.median()
    ymedian = ycol.median()
    plt.axvline(x=xmedian, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.axhline(y=ymedian, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Medián szövegek
    plt.text(xmedian + 0.01, plt.ylim()[0] + 0.01, 'Median', 
             c='white', alpha=0.8, fontsize=10, fontweight='bold')
    plt.text(plt.xlim()[0] + 0.01, ymedian + 0.01, 'Median', 
             c='white', alpha=0.8, fontsize=10, fontweight='bold')
    
    # Styling
    plt.xlabel(xlabel, fontsize=14, color='white', fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, color='white', fontweight='bold')
    plt.title(title, fontsize=20, color='white', fontweight='bold', pad=20)
    
    # Axis styling
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_visible(False)
    
    # Grid and ticks
    plt.grid(True, color='#555555', alpha=0.3, linestyle='--')
    plt.tick_params(colors='white')
    
    # Add signature
    fig.text(0.88, -0.00, 'ADAM JAKUS', color=mycolor, fontsize=16, 
              ha='center', va='center')
    
    # Add league info
    league_name = df_super['league'].iloc[0] if 'league' in df_super.columns else league
    fig.text(0.10, 0.95, f'{league_name} | {only_position}', color='white', 
             fontsize=12, fontweight='normal', ha='left', va='center')
    
    # Add min 90s played info
    fig.text(0.10, 0.92, f'Min {min_90_played} games played', color='#AAAAAA', 
             fontsize=10, fontweight='normal', ha='left', va='center')
    
    plt.tight_layout()
    
    # Show plot
    plt.show()

#%% 
# Pass nr. - Pass comp% -> Total_Att - Total_Cmp%
data_to_scatter('Total_Att_p90', 'Total_Cmp%', 'Attempted passes per90', 'Pass completion (%)', 'Passing', ids=[393])
# Shots nr.  - shot% -> Standard_Sh_p90 - ShConversion%
df_super.drop(index= df_super[df_super.Standard_Sh_p90 == 0].index, inplace=True)
df_super['ShConversion%'] = (df_super['Standard_Gls_p90'] / df_super['Standard_Sh_p90'])*100
data_to_scatter('Standard_Sh_p90', 'ShConversion%', 'Shots per90', 'Shot conversion (%)', 'Shooting', ids=[393])
data_to_scatter('Touches_Touches_p90', 'GCA_GCA_p90', 'Touches per 90', 'Goal creating actions per 90', 'Activity', ids=[393])
data_to_scatter('Progression_PrgC_p90', 'Progression_PrgP_p90', 'Progressive Carries per 90', 'Progressive Passes per 90', 'Progression', ids=[393])

[col for col in df_super.columns if 'Prg' in col]
