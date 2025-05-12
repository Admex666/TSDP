#%% Scrape the data from fbref
import pandas as pd
import numpy as np
from fbref import fbref_module as fbref
import matplotlib.pyplot as plt

# Set parameters
league_name = 'Big5'

dataframe = fbref.get_all_team_data(league_name, year=False)

#%% Format the merged data a bit
df = dataframe.copy()
base_cols = ['Rk', 'Squad', 'Comp', '# Pl', 'Age', 'Poss', '90s'] + [col for col in df.columns if 'Playing Time' in col]

df90 = df.copy()
for col in df90.drop(columns=base_cols).columns:
    if ('90' in col) or ('%' in col) or ('Playing Time' in col):
        pass
    else:
        df90[col] = df90[col]/df90['90s']
        df90.rename(columns={col:f'{col}_p90'}, inplace=True)

#%% To excel
path = 'fbref/squad_scatters.xlsx'
df90.to_excel(path, index=False)

#%% Plotting (defining function)

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

def data_to_scatter(df_super, xname, yname, xlabel, ylabel, title):
    xcol = df_super[xname]
    ycol = df_super[yname]
    
    np.random.seed(1)
    colors = np.random.rand(len(df_super.Comp), 3)  # Generate random RGB colors for each team
    league_color_dict = dict(zip(df_super.Comp, colors))
    #team_color_dict = la_liga_team_colors
    
    plt.figure(figsize=(12, 8))
    for league, color in league_color_dict.items():
        # Filter data for the current team
        league_data = df_super[df_super['Comp'] == league]
        # Plot with the team's color and label
        xdata = league_data[xname]
        ydata = league_data[yname]    
        plt.scatter(
            x=xdata,
            y=ydata,
            color=color,
            label=league,
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
    
    # Identify outliers and add team names
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
            s=row['Squad'],  # Player name
            fontsize=9,
            color='black',
            ha='left',  # Horizontal alignment
            va='bottom'  # Vertical alignment
        )
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc="lower right")
    plt.title(title, fontsize=24)
    plt.show()

#%% 
# Pass nr. - Pass comp% -> Total_Att - Total_Cmp%
data_to_scatter(df90, 'Total_Att_p90', 'Total_Cmp%', 'Attempted passes per 90', 'Pass completion (%)', 'Passing')
# Shots nr.  - shot% -> Standard_Sh_p90 - ShConversion%
df_super.drop(index= df_super[df_super.Standard_Sh_p90 == 0].index, inplace=True)
df_super['ShConversion%'] = (df_super['Standard_Gls_p90'] / df_super['Standard_Sh_p90'])*100
data_to_scatter('Standard_Sh_p90', 'ShConversion%', 'Shots per90', 'Shot conversion (%)', 'Shooting')
"""