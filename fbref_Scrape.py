import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%% dynamic URL
def fbref_input_season(season):
    URL = 'https://fbref.com/en/comps/20/'+ season + '/' + season + '-Bundesliga-Stats'
    return URL
            
season = '2019-2020'
URL = fbref_input_season(season)

###URL = 'https://fbref.com/en/comps/20/2019-2020/2019-2020-Bundesliga-Stats'

#%% Understanding the logic of fbref
"""
df_league_table = pd.read_html(URL,
                  attrs={"id":"results2022-2023201_overall"})
# if df is a list, including dataframe at index 0
df_league_table = df_league_table[0]

# if df is in DataFrame type, but with lot of NaN
### df = df.dropna(subset=['Rk'])
"""
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
    if type(df.columns.values[0]) != str:
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    else:
        return df

def fbref_format_column_names(df):
    for ncol in range(len(df.columns)):
        if 'Unnamed' in df.columns[ncol]:
            col_old_name = df.columns[ncol]
            col_new_name = df.columns[ncol].split('_')[3]
            df.rename(columns={col_old_name:col_new_name}, inplace=True)
        else:
            pass

def fbref_scrape(URL, table_id):
    df = fbref_column_joiner(fbref_to_dataframe(fbref_read_html(URL, table_id)))
    return df

def fbref_season_to_next(season):
    season_list = season.split('-')
    season_list[0] = str(int(season_list[0])+1)
    season_list[1] = str(int(season_list[1])+1)
    new_season = '-'.join(season_list)
    return new_season

#%% importing dataframes
df_goalkeeping = fbref_column_joiner(
    fbref_to_dataframe(
    fbref_read_html(URL, 'stats_squads_keeper_for')))

df_shooting_for = fbref_scrape(URL, 'stats_squads_shooting_for')
df_shooting_against = fbref_scrape(URL, 'stats_squads_shooting_against')
df_league_table = fbref_scrape(URL, 'results'+ season +'201_overall')

#%% Formatting columns
fbref_format_column_names(df_goalkeeping)
fbref_format_column_names(df_shooting_for)
fbref_format_column_names(df_shooting_against)

df_league_table.sort_values(by='Squad', ascending=True, inplace=True)
df_league_table['Squad_ID'] = range(18)
df_shooting_for['Squad_ID'] = range(18)
df_shooting_against['Squad_ID'] = range(18)

## changing 'vs' in names
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

df_reg.sort_values(by='Rk', inplace=True)
## need to calculate Shot-ShotAgainst column
df_reg['ShotDiff'] = df_reg.Standard_Sh_For - df_reg.Standard_Sh_Against

#%% need to calculate coeff of determination, and beta parameters
def fbref_linreg(col1, col2):
    model = LinearRegression()
    array_col1 = np.array([col1]).reshape((-1,1))
    
    model.fit(array_col1, col2)
    
    r2 = model.score(array_col1, col2)
    b0 = model.intercept_
    b1 = model.coef_[0]
    return [r2, b0, b1]

r2_reg = fbref_linreg(df_reg.ShotDiff, df_reg.GD)[0]
b0_reg = fbref_linreg(df_reg.ShotDiff, df_reg.GD)[1]
b1_reg = fbref_linreg(df_reg.ShotDiff, df_reg.GD)[2]

## predicted values of the model
### GD - GD_est = GD_est_diff
df_reg['GD_est_diff'] = df_reg['GD'] - (b0_reg + b1_reg * df_reg.ShotDiff)

#%% Importing next season's(2) points to check the change
season2 = fbref_season_to_next(season)
URL2 = fbref_input_season(season2)

df_league_table2 = fbref_scrape(URL2,
                                   'results'+ season2 +'201_overall')
df_reg = pd.merge(df_reg, df_league_table2[['Squad', 'Pts']], 
                  how='inner', on='Squad')
df_reg.rename(columns={'Pts_x':'Pts','Pts_y':'Pts2'}, inplace=True)
df_reg['Pts_change'] = df_reg.Pts2 - df_reg.Pts

#%% calculate r2, b0 and b1 for Exp_GD(1) & Points change(2-1)
model_ptch = LinearRegression()
array_gd_reg = np.array([df_reg.GD_est_diff]).reshape((-1,1))

model_ptch.fit(array_gd_reg, df_reg.Pts_change)

r2_ptch = model_ptch.score(array_gd_reg, df_reg.Pts_change)
b0_ptch = model_ptch.intercept_
b1_ptch = model_ptch.coef_[0]

#%% Do the same to estimate points interval for the next season(3)
## import next year's(2) tables, and format them
df_shooting_for2 = fbref_scrape(URL2, 'stats_squads_shooting_for')
df_shooting_against2 = fbref_scrape(URL2, 'stats_squads_shooting_against')

df_league_table2.sort_values(by='Squad', ascending=True, inplace=True)

tables_list2 = [df_league_table2, df_shooting_for2, df_shooting_against2] 

for dfs in range(len(tables_list2)):
    fbref_format_column_names(tables_list2[dfs])
    tables_list2[dfs]['Squad_ID'] = range(len(tables_list2[dfs]))

df_league_table2 = tables_list2[0]
df_shooting_for2 = tables_list2[1]
df_shooting_against2 = tables_list2[2]

for sq in range(len(df_shooting_against2.Squad)):
    df_shooting_against2.Squad[sq] = df_shooting_against2.Squad[sq].split(' ')[1]

## regression table
df_reg2 = pd.merge(df_league_table2[['Squad_ID','Rk', 'Squad', 'GD', 'Pts']],
                  df_shooting_for2[['Squad_ID','Standard_Sh']], 
                  on='Squad_ID')
df_reg2 = pd.merge(df_reg2,
                  df_shooting_against2[['Squad_ID', 'Standard_Sh']],
                  on='Squad_ID')

df_reg2.rename(columns={'Standard_Sh_x':'Standard_Sh_For',
                       'Standard_Sh_y':'Standard_Sh_Against'},
              inplace=True)
df_reg2.sort_values(by='Rk', inplace=True)
## need to calculate Shot-ShotAgainst column
df_reg2['ShotDiff'] = df_reg2.Standard_Sh_For - df_reg2.Standard_Sh_Against

## estimate the points change from exp_GD(2) to Points_change(3-2)
r2_reg2 = fbref_linreg(df_reg2.ShotDiff, df_reg2.GD)[0]
b0_reg2 = fbref_linreg(df_reg2.ShotDiff, df_reg2.GD)[1]
b1_reg2 = fbref_linreg(df_reg2.ShotDiff, df_reg2.GD)[2]

df_reg2['GD_est_diff'] = df_reg2['GD'] - (b0_reg2 + b1_reg2 * df_reg2.ShotDiff)

#%% Estimate the values and intervals
## estimate with the parameters of the previous season()
df_reg2['est_Pts3_change'] = b0_ptch + df_reg2.GD_est_diff*b1_ptch
df_reg2['est_Pts3'] = df_reg2.Pts + df_reg2.est_Pts3_change

## stats
import scipy.stats as stats

ptch_alpha = 0.85
ptch_t_value = stats.t.ppf((1.0+ptch_alpha) / 2, len(df_reg)-1)
ptch_std = np.std(df_reg.Pts_change, ddof=1)
ptch_delta = ptch_t_value * ptch_std

df_reg2['est_Pts3_min'] = round((df_reg2.Pts - ptch_delta),0)
df_reg2['est_Pts3_max'] = round((df_reg2.Pts + ptch_delta),0)

#%% Check exp_season3 and fact_season3
season3 = fbref_season_to_next(season2)
URL3 = fbref_input_season(season3)

df_league_table3 = fbref_scrape(URL3,
                                   'results'+ season3 +'201_overall')

df_comparison = pd.merge(df_reg2[['Squad', 'est_Pts3', 'est_Pts3_min', 'est_Pts3_max']],
                         df_league_table3[['Squad','Pts']],
                         how='inner', on='Squad')

df_comparison['abs_PtsDiff'] = abs(df_comparison.Pts - df_comparison.est_Pts3)
print('The average error:', np.average(df_comparison.abs_PtsDiff))

df_comparison['isIn_interval'] = 100
for x in range(len(df_comparison)):
    if (df_comparison.Pts[x] >= df_comparison.est_Pts3_min[x]) & (df_comparison.Pts[x] <= df_comparison.est_Pts3_max[x]):
        df_comparison['isIn_interval'][x] = 1
    else:
        df_comparison['isIn_interval'][x] = 0
        
df_comparison.isIn_interval.sum()/len(df_comparison)
