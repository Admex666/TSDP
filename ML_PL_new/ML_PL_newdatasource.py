import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import fbref_module as fbref

# Loading data from website
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df = df[needed_cols]
# create BTTS and O2,5 labels
df['BTTS'] = np.where((df.FTHG!=0)&(df.FTAG!=0),'Yes','No')
df['O/U2.5'] = np.where(df.FTHG+df.FTAG>2.5,'Over','Under')

#%% Transforming data
def df_to_model_input(df):  
    # Writing it out
    df.rename(columns={
        'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals',
        'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget', 'AST': 'AwayShotsOnTarget',
        'HC': 'HomeCorners', 'AC': 'AwayCorners',
        'HY': 'HomeYellows', 'AY': 'AwayYellows',
    }, inplace=True)
    
    # Date form
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values('Date', inplace=True)
    
    # Stats list for rolling averages
    stats = ['Goals', 'Shots', 'ShotsOnTarget', 'Corners', 'Yellows']
    home_stats = ['Home' + stat for stat in stats]
    away_stats = ['Away' + stat for stat in stats]
    
    # Rolling average calculations for each team
    rolling_features = []
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        team_home = df[df['HomeTeam'] == team].copy()
        team_away = df[df['AwayTeam'] == team].copy()
    
        for stat, home_stat, away_stat in zip(stats, home_stats, away_stats):
            team_home[stat + '_Home_RAvg'] = team_home[home_stat].shift(1).rolling(window=3).mean()
            team_away[stat + '_Away_RAvg'] = team_away[away_stat].shift(1).rolling(window=3).mean()
    
        rolling_features.append(pd.concat([team_home, team_away]))
    
    # Summing rolling averages
    rolling_df = pd.concat(rolling_features).sort_values(['Date', 'HomeTeam'])
    
    final_df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'BTTS', 'O/U2.5']].copy()
    
    # Home team rolling average merge
    home_rolling = rolling_df[['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]]
    home_rolling.columns = ['Date', 'HomeTeam'] + [stat + '_Home_RAvg' for stat in stats]
    final_df = final_df.merge(home_rolling, on=['Date', 'HomeTeam'])
    
    # Away team rolling average merge
    away_rolling = rolling_df[['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats]]
    away_rolling.columns = ['Date', 'AwayTeam'] + [stat + '_Away_RAvg' for stat in stats]
    final_df = final_df.merge(away_rolling, on=['Date', 'AwayTeam'], how='left')
    
    # Final data structure
    model_input = final_df.copy().dropna().reset_index(drop=True)
    return model_input

model_input = df_to_model_input(df)

#%% Build ML model
df_accs = pd.DataFrame()

params_grid = {'GaussianNB':{},
               'DecisionTreeClassifier': {'max_depth': 4, 'min_samples_split': 6,
                                          'min_samples_leaf': 2, 'criterion':'entropy'
                                          },
               'RandomForestClassifier': {'max_depth': 7, 'min_samples_split': 5,
                                          'min_samples_leaf': 2, 'max_features': 'sqrt', 
                                          'bootstrap': True      
                                          }
               }
model_short_dict = {'GaussianNB':'gNB', 
                    'RandomForestClassifier': 'RF',
                    'DecisionTreeClassifier': 'DT'}


for btype in ['FTR', 'BTTS', 'O/U2.5']:
    x = model_input.iloc[:,6:]
    y = model_input.loc[:, btype]
    for m in ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        acc_list = []
        m_short = model_short_dict[m]
        for n in range(170, 220):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=n)
            
            model = globals()[f"{m}"]()
            model.set_params(**params_grid[m])
            
            model.fit(x_train, y_train)
            
# Evaluation
            y_pred = model.predict(x_test)
            accuracy= accuracy_score(y_pred, y_test)
            acc_list.append(accuracy)
        df_accs[f'{btype}_{m_short}'] = acc_list
    
accs_describe = df_accs.describe().iloc[1:, :]
df_accs.boxplot(rot=90)
    
#%% Optimizing parameters
"""
params_all = list(model.get_params().keys())
params = {
    'max_depth': [None, 5, 7, 10, 12, 15, 17, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
rsearch = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch.best_score_)
rs_results = pd.DataFrame(rsearch.cv_results_)
"""
#%% Getting the fresh data for predictions
import pandas as pd
import numpy as np
from thefuzz import fuzz
path_pred = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
df_pred = pd.read_csv(path_pred)

#%% Transform
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df_pred = df_pred[needed_cols]
# create BTTS and O2,5 labels
df_pred['BTTS'] = np.where((df_pred.FTHG!=0)&(df_pred.FTAG!=0),'Yes','No')
df_pred['O/U2.5'] = np.where(df_pred.FTHG+df_pred.FTAG>2.5,'Over','Under')

teams = np.sort(df_pred.HomeTeam.unique())

# Add current matches
select_date = '22/02/2025'

#%% Create next round's pairings
df_current = df_pred.copy()
df_current = df_current.iloc[:10]
df_current.Date = select_date
df_current.iloc[:,3:-3] = 0
df_current[['FTR','BTTS','O/U2.5']] = 'none'

comp_id, league = fbref.team_dict_get('ENG')
url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
df_fixtures = fbref.scrape(url_fixtures, 'sched_2024-2025_9_1')
mask = (df_fixtures.Wk != 'Wk') & (df_fixtures.Score.isna()) & (df_fixtures.Wk.notna())
weeknr = df_fixtures.loc[mask,:].reset_index(drop=True).loc[0,'Wk']
df_week = df_fixtures.loc[df_fixtures.Wk == weeknr, :].reset_index(drop=True)

fuzz_teams = pd.DataFrame({'Team': teams,
                           'fbref_team': None,
                           'ratio': None,
                           'home_away':None,
                           'matchnr':None}
                          )
for i, team in enumerate(teams):
    highest_ratio = 0
    for fbrteam in [*df_week['Home'].unique(), *df_week['Away'].unique()]:
        ratio = fuzz.ratio(team, fbrteam)
        if fbrteam in df_week['Home'].unique():
            home_away = 'Home'
        else:
            home_away = 'Away'
        
        if ratio > highest_ratio:
            highest_ratio = ratio
            highest_team = fbrteam
            highest_ha = home_away
            match_nr = df_week.loc[df_week[highest_ha] == highest_team, :].index[0]
            
    #print(i, home)
    fuzz_teams.loc[i, 'fbref_team'] = highest_team
    fuzz_teams.loc[i, 'ratio'] = highest_ratio
    fuzz_teams.loc[i, 'home_away'] = highest_ha
    fuzz_teams.loc[i, 'matchnr'] = match_nr

for x in range(10):
    mask_home = (fuzz_teams.matchnr == x) & (fuzz_teams.home_away == 'Home')
    mask_away = (fuzz_teams.matchnr == x) & (fuzz_teams.home_away == 'Away')
    df_current.loc[x, ['HomeTeam', 'AwayTeam']] = [fuzz_teams.loc[mask_home,'Team'].iloc[0],
                                                   fuzz_teams.loc[mask_away,'Team'].iloc[0]]

df_all = pd.concat([df_pred,df_current], ignore_index=True)

model_input_pred = df_to_model_input(df_all)
model_input_pred = model_input_pred.iloc[-10:,:].reset_index(drop=True)

#%% Build model
predictions = model_input_pred[['HomeTeam', 'AwayTeam']].copy()
for btype in ['FTR', 'BTTS', 'O/U2.5']:
    x_test = model_input_pred.iloc[:,6:]
    # train on previous season
    x_train = model_input.iloc[:,6:]
    y_train = model_input.loc[:, btype]
    for m in ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        m_short = model_short_dict[m]
        
        model = globals()[f"{m}"]()
        model.set_params(**params_grid[m])
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        predictions[f'{btype}_{m_short}'] = y_pred