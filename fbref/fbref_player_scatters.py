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
import fbref_module as fbref
import matplotlib.pyplot as plt

# Set parameters
league = 'UEL'
min_90_played = 0 # how many matches at least
only_position = 'MF' # DF, MF, FW or GK

team_dict = {'ENG': {'comp_id':'9', 'league':'Premier-League'},
             'ESP': {'comp_id':'12', 'league':'La-Liga'},
             'GER': {'comp_id':'20', 'league':'Bundesliga'},
             'ITA': {'comp_id':'11', 'league':'Serie-A'},
             'UEL': {'comp_id':'19', 'league':'Europa-League'}}

comp_id = team_dict.get(league).get('comp_id')
league_name = team_dict.get(league).get('league')

urllist = ['passing', 'shooting', 'defense', 'misc', 'keepers', 'keepersadv']
#URLs
URL_standard = ("https://fbref.com/en/comps/" + comp_id + '/stats/'
                + league_name + '-Stats#all_stats_' + 'standard'
                )
for stat in urllist:
    globals()[f'URL_{stat}'] = ("https://fbref.com/en/comps/" + comp_id
                                + '/{}/'.format(stat) + league_name
                                + '-Stats#all_stats_' + stat
                                )

del comp_id, league

#%% Preparing df and functions for scrape
def scrape(URL, table_id):       
    df = fbref.read_html_upd(URL, table_id)
    df = df[0]
    df = fbref.format_column_names(fbref.column_joiner(df))
    df = df.iloc[:-1,:]
    return df

#%% Scraping the actual data
urllist.append('standard')
statlist = ['passing', 'shooting', 'defense', 'misc', 'keeper', 'keeper_adv', 'standard']

for stat, url in zip(statlist, urllist):
    globals()[f'df_{stat}'] = scrape(globals()[f'URL_{url}'], f'stats_{stat}')
    # remove header rows
    ind = globals()[f'df_{stat}'].loc[globals()[f'df_{stat}'].Rk=='Rk', 'Rk'].index
    globals()[f'df_{stat}'].drop(index=ind, inplace=True)
    globals()[f'df_{stat}'].reset_index(drop=True, inplace=True)
    
df_keeper.rename(columns={'Performance_SoTA': 'SoT_against'}, inplace=True)

#%% Create a merged dataframe
df_super = df_passing.copy()

for stat in statlist[1:]:
    df_super = pd.merge(df_super, globals()[f'df_{stat}'], 
                        on=['Player', 'Squad'], how='outer', 
                        suffixes=('', '_remove'))
    df_super.drop([i for i in df_super.columns if 'remove' in i], 
                  axis=1, inplace=True)

# Filter by parameters
df_super = df_super.loc[df_super['90s'].astype(float) > min_90_played, :]
df_super = df_super.loc[df_super['Pos'].str.contains(only_position), :]

#%% Format the merged data a bit
df_super.drop(columns='Matches', inplace=True)
df_super.iloc[:,6:] = df_super.iloc[:,6:].astype(float) # set dtype
df_super.Age = df_super.Age.str.split('-').str.get(0).astype(float) # get age as number
df_super.drop(index=df_super.loc[df_super['90s'] == 0].index, inplace=True)

for col in df_super.columns[8:]:
    if ('90' in col) or ('%' in col):
        pass
    else:
        df_super[col] = df_super[col]/df_super['90s']
        df_super.rename(columns={col:f'{col}_p90'}, inplace=True)

#%% To excel
path = r'C:\TSDP\fbref\player_scatters.xlsx'
df_super.to_excel(path, index=False)

#%% Plotting (defining function)
"""
la_liga_team_colors = {
    "Alavés": (10/255, 63/255, 245/255),  # Blue
    "Athletic Club": (238/255, 37/255, 35/255),  # Red
    "Atlético Madrid": (206/255, 53/255, 36/255),  # Red
    "Barcelona": (0/255, 77/255, 152/255),  # Blue
    "Celta Vigo": (138/255, 195/255, 238/255),  # Light Blue
    "Espanyol": (30/255, 107/255, 192/255),  # Blue
    "Getafe": (0/255, 77/255, 152/255),  # Blue
    "Girona": (200/255, 16/255, 46/255),  # Red
    "Las Palmas": (255/255, 204/255, 0/255),  # Yellow
    "Leganés": (0/255, 128/255, 0/255),  # Green
    "Mallorca": (227/255, 27/255, 35/255),  # Red
    "Osasuna": (0/255, 0/255, 191/255),  # Blue
    "Betis": (0/255, 149/255, 76/255),  # Green
    "Real Madrid": (255/255, 255/255, 255/255),  # White
    "Real Sociedad": (0/255, 77/255, 152/255),  # Blue
    "Sevilla": (218/255, 41/255, 28/255),  # Red
    "Valencia": (251/255, 181/255, 18/255),  # Yellow
    "Villarreal": (250/255, 220/255, 0/255),  # Yellow
    "Rayo Vallecano": (229/255, 48/255, 41/255),  # Red
    "Valladolid": (146/255, 30/255, 127/255),  # Purple
}

def data_to_scatter(xname, yname, xlabel, ylabel, title):
    xcol = df_super[xname]
    ycol = df_super[yname]
    
    #np.random.seed(10)
    #colors = np.random.rand(len(df_super.Squad), 3)  # Generate random RGB colors for each team
    #team_color_dict = dict(zip(df_super.Squad, colors))
    team_color_dict = la_liga_team_colors
    
    plt.figure(figsize=(12, 8))
    for team, color in team_color_dict.items():
        # Filter data for the current team
        team_data = df_super[df_super['Squad'] == team]
        # Plot with the team's color and label
        xdata = team_data[xname]
        ydata = team_data[yname]    
        plt.scatter(
            x=xdata,
            y=ydata,
            color=color,
            label=team,
            alpha=1
        )
    
    # Plot median lines
    xmedian = xcol.median()
    ymedian = ycol.median()
    plt.axvline(x=xmedian, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=ymedian, color='blue', linestyle='--', alpha=0.5)
    
    # Text next to median lines
    plt.text(xmedian + 0.1, plt.ylim()[0] + 0.1, 'Median', c='blue', alpha=0.5)
    plt.text(plt.xlim()[0] + 0.1, ymedian + 0.1, 'Median', c='blue', alpha=0.5, ha='left', va='bottom')
    
    # Identify outliers and add player names
    def find_outliers(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (column < lower_bound) | (column > upper_bound)
    
    x_outliers = find_outliers(xcol)
    y_outliers = find_outliers(ycol)
    outliers = x_outliers | y_outliers  # Combine x and y outliers
    
    # Add text labels for outliers
    for i, row in df_super[outliers].iterrows():
        plt.text(
            x=row[xname] + 0.1,  # Offset text slightly to the right of the point
            y=row[yname] + 0.1,  # Offset text slightly above the point
            s=row['Player'],  # Player name
            fontsize=9,
            color='black',
            ha='left',  # Horizontal alignment
            va='bottom'  # Vertical alignment
        )
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    #plt.legend(loc="lower right")
    plt.title(title, fontsize=24)
    plt.show()

#%% 
# Pass nr. - Pass comp% -> Total_Att - Total_Cmp%
data_to_scatter('Total_Att_p90', 'Total_Cmp%', 'Attempted passes per90', 'Pass completion (%)', 'Passing')
# Shots nr.  - shot% -> Standard_Sh_p90 - ShConversion%
df_super.drop(index= df_super[df_super.Standard_Sh_p90 == 0].index, inplace=True)
df_super['ShConversion%'] = (df_super['Standard_Gls_p90'] / df_super['Standard_Sh_p90'])*100
data_to_scatter('Standard_Sh_p90', 'ShConversion%', 'Shots per90', 'Shot conversion (%)', 'Shooting')
"""