import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
URL = 'https://fbref.com/en/comps/20/2022-2023/2022-2023-Bundesliga-Stats'

#%% Understanding the logic of fbref
df_league_table = pd.read_html(URL,
                  attrs={"id":"results2022-2023201_overall"})
# if df is a list, including dataframe at index 0
df_league_table = df_league_table[0]

# if df is in DataFrame type, but with lot of NaN
### df = df.dropna(subset=['Rk'])

#%% creating a function for reading HTMLs
def fbref_read_html(URL, table_id):
    df = pd.read_html(URL, attrs={"id": table_id})
    return df

# creating a function for transforming scraped data into proper dataframes
def fbref_to_dataframe(df):
    if (type(df) == list) & (len(df)==1):
        df = df[0]
    elif type(df) == pd.core.frame.DataFrame:
        df = df.dropna(subset=['Rk'])
    else:
        print('Unknown df type')
        
    if type(df) != pd.core.frame.DataFrame:
        print('Not a pandas dataframe')
    else:
        return df

def fbref_column_joiner(df):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

def fbref_scrape(URL, table_id):
    df = fbref_column_joiner(fbref_to_dataframe(fbref_read_html(URL, table_id)))
    return df

#%% importing dataframes
df_goalkeeping = fbref_column_joiner(
    fbref_to_dataframe(
    fbref_read_html(URL, 'stats_squads_keeper_for')))

df_shooting_for = fbref_scrape(URL, 'stats_squads_shooting_for')
df_shooting_against = fbref_scrape(URL, 'stats_squads_shooting_against')

#%% Formatting columns
def fbref_format_column_names(df):
    for ncol in range(len(df.columns)):
        if 'Unnamed' in df.columns[ncol]:
            col_old_name = df.columns[ncol]
            col_new_name = df.columns[ncol].split('_')[3]
            df.rename(columns={col_old_name:col_new_name}, inplace=True)
        else:
            pass

fbref_format_column_names(df_goalkeeping)
fbref_format_column_names(df_shooting_for)
fbref_format_column_names(df_shooting_against)

df_league_table.sort_values(by='Squad', ascending=True, inplace=True)
df_league_table['Squad_ID'] = range(18)
df_shooting_for['Squad_ID'] = range(18)
df_shooting_against['Squad_ID'] = range(18)

for sq in range(len(df_shooting_against.Squad)):
    df_shooting_against.Squad[sq] = df_shooting_against.Squad[sq].split(' ')[1]
#%% Creating the table for ShotDiff - GoalDiff regression
## need Squad -merge on, league_table(Rk, GD, Pts,) shooting_for(Shot,) (Shot against)
df_reg = pd.merge(df_league_table[['Squad_ID','Rk', 'Squad', 'GD', 'Pts']],
                  df_shooting_for[['Squad_ID','Standard_Sh']], 
                  on='Squad_ID')
df_reg = pd.merge(df_reg,
                  df_shooting_against[['Squad_ID', 'Standard_Sh']],
                  on='Squad_ID')

df_reg.rename(columns={'Standard_Sh_x':'Standard_Sh_For',
                       'Standard_Sh_y':'Standard_Sh_Against'},
              inplace=True)

## need to calculate Shot-ShotAgainst column
df_reg['ShotDiff'] = df_reg.Standard_Sh_For - df_reg.Standard_Sh_Against

#%% need to calculate coeff of determination, and beta parameters
model = LinearRegression()
array_sd = np.array([df_reg.ShotDiff]).reshape((-1,1))

model.fit(array_sd, df_reg.GD)

r2 = model.score(array_sd, df_reg.GD)
b0 = model.intercept_
b1 = model.coef_[0]

## predicted values of the model
### GD - GD_est = GD_est_diff
df_reg['GD_est_diff'] = df_reg['GD'] - (b0 + b1 * df_reg.ShotDiff)
