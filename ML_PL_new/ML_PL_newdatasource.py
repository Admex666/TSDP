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

# Prepare ML model
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

#%% Build ML model
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

# football-data.co.uk historical data urls:
csv_name_dict = {'ENG':'E0',
                 'ESP':'SP1', 
                 'GER': 'D1', 
                 'ITA': 'I1',
                 'FRA': 'F1'}

predictions_merged = pd.DataFrame()
#fuzz_teams_merged = pd.DataFrame()
for countrycode in ['ENG', 'ESP', 'GER', 'ITA', 'FRA']:
    csv_name = csv_name_dict[countrycode]

    path_pred = f"https://www.football-data.co.uk/mmz4281/2425/{csv_name}.csv"
    df_pred = pd.read_csv(path_pred)
    
    # Transform
    needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
    df_pred = df_pred[needed_cols]
    # create BTTS and O2,5 labels
    df_pred['BTTS'] = np.where((df_pred.FTHG!=0)&(df_pred.FTAG!=0),'Yes','No')
    df_pred['O/U2.5'] = np.where(df_pred.FTHG+df_pred.FTAG>2.5,'Over','Under')
    
    teams = np.sort(df_pred.HomeTeam.unique())
    nr_matches = int(len(teams)/2)
    
    # Create next round's pairings
    df_current = df_pred.copy()
    df_current = df_current.iloc[:nr_matches]
    df_current.iloc[:,3:-3] = 0
    df_current[['FTR','BTTS','O/U2.5']] = 'none'
    
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    mask = (df_fixtures.Wk != 'Wk') & (df_fixtures.Score.isna()) & (df_fixtures.Wk.notna())
    weeknr = df_fixtures.loc[mask,:].reset_index(drop=True).loc[0,'Wk']
    df_week = df_fixtures.loc[df_fixtures.Wk == weeknr, :].reset_index(drop=True)

    # Fuzzy matched squads list:
    teams_fdcouk = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
           'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
           'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
           "Nott'm Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves',
           'Alaves', 'Ath Bilbao', 'Ath Madrid', 'Barcelona', 'Betis',
           'Celta', 'Espanol', 'Getafe', 'Girona', 'Las Palmas', 'Leganes',
           'Mallorca', 'Osasuna', 'Real Madrid', 'Sevilla', 'Sociedad',
           'Valencia', 'Valladolid', 'Vallecano', 'Villarreal', 'Augsburg',
           'Bayern Munich', 'Bochum', 'Dortmund', 'Ein Frankfurt', 'Freiburg',
           'Heidenheim', 'Hoffenheim', 'Holstein Kiel', 'Leverkusen',
           "M'gladbach", 'Mainz', 'RB Leipzig', 'St Pauli', 'Stuttgart',
           'Union Berlin', 'Werder Bremen', 'Wolfsburg', 'Atalanta',
           'Bologna', 'Cagliari', 'Como', 'Empoli', 'Fiorentina', 'Genoa',
           'Inter', 'Juventus', 'Lazio', 'Lecce', 'Milan', 'Monza', 'Napoli',
           'Parma', 'Roma', 'Torino', 'Udinese', 'Venezia', 'Verona',
           'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens', 'Lille', 'Lyon',
           'Marseille', 'Monaco', 'Montpellier', 'Nantes', 'Nice', 'Paris SG',
           'Reims', 'Rennes', 'St Etienne', 'Strasbourg', 'Toulouse']
    teams_fbref = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
           'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
           'Leicester City', 'Liverpool', 'Manchester City', 'Manchester Utd',
           'Newcastle Utd', "Nott'ham Forest", 'Southampton', 'Tottenham',
           'West Ham', 'Wolves', 'Alavés', 'Athletic Club', 'Atlético Madrid',
           'Barcelona', 'Betis', 'Celta Vigo', 'Espanyol', 'Getafe', 'Girona',
           'Las Palmas', 'Leganés', 'Mallorca', 'Osasuna', 'Real Madrid',
           'Sevilla', 'Real Sociedad', 'Valencia', 'Valladolid',
           'Rayo Vallecano', 'Villarreal', 'Augsburg', 'Bayern Munich',
           'Bochum', 'Dortmund', 'Eint Frankfurt', 'Freiburg', 'Heidenheim',
           'Hoffenheim', 'Holstein Kiel', 'Leverkusen', 'Gladbach',
           'Mainz 05', 'RB Leipzig', 'St. Pauli', 'Stuttgart', 'Union Berlin',
           'Werder Bremen', 'Wolfsburg', 'Atalanta', 'Bologna', 'Cagliari',
           'Como', 'Empoli', 'Fiorentina', 'Genoa', 'Inter', 'Juventus',
           'Lazio', 'Lecce', 'Milan', 'Monza', 'Napoli', 'Parma', 'Roma',
           'Torino', 'Udinese', 'Venezia', 'Hellas Verona', 'Angers',
           'Auxerre', 'Brest', 'Le Havre', 'Lens', 'Lille', 'Lyon',
           'Marseille', 'Monaco', 'Montpellier', 'Nantes', 'Nice',
           'Paris S-G', 'Reims', 'Rennes', 'Saint-Étienne', 'Strasbourg',
           'Toulouse']
    fuzz_teams_all = pd.DataFrame({'Team_fdcouk':teams_fdcouk, 'Team_fbref':teams_fbref})
    fuzz_teams_all['Country']=[*['ENG']*20,*['ESP']*20,*['GER']*18,*['ITA']*20,*['FRA']*18]
    fuzz_teams = fuzz_teams_all[fuzz_teams_all.Country == countrycode]
   
    for fbrteam in fuzz_teams.Team_fbref:
        i = fuzz_teams[fuzz_teams.Team_fbref == fbrteam].index[0]
        if fbrteam in df_week['Home'].unique():
            home_away = 'Home'
        elif fbrteam in df_week['Away'].unique():
            home_away = 'Away'
        else:
            home_away = 'ERROR'
        
        fuzz_teams.loc[i, 'home_away'] = home_away
        fuzz_teams.loc[i, 'matchnr'] = df_week.loc[df_week[home_away] == fbrteam, :].index[0]
    
    for x in range(nr_matches):
        mask_home = (fuzz_teams.matchnr == x) & (fuzz_teams.home_away == 'Home')
        mask_away = (fuzz_teams.matchnr == x) & (fuzz_teams.home_away == 'Away')
        df_current.loc[x, ['HomeTeam', 'AwayTeam']] = [fuzz_teams.loc[mask_home,'Team_fdcouk'].iloc[0],
                                                       fuzz_teams.loc[mask_away,'Team_fdcouk'].iloc[0]]
        df_current.loc[x, 'Date'] = df_week.loc[x, 'Date']
    df_current['Date'] = pd.to_datetime(df_current['Date'], format='%Y-%m-%d')
    
    df_all = pd.concat([df_pred,df_current], ignore_index=True)
    
    model_input_pred = df_to_model_input(df_all)
    model_input_pred = model_input_pred.iloc[-nr_matches:,:].reset_index(drop=True)
    
    # Build model
    predictions = model_input_pred[['Date','HomeTeam', 'AwayTeam']].copy()
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
            
            proba = model.predict_proba(x_test)
            classes = model.classes_
            for i, clss in enumerate(classes):
                predictions[f'{clss}_{m_short}_prob'] = proba[:, i]
        
    predictions_merged = pd.concat([predictions_merged, predictions])
    
#%% To excel
predictions_merged = predictions_merged.sort_values(by='Date').reset_index(drop=True)
predictions_merged.to_excel(r'C:\Users\Ádám\Dropbox\TSDP_output\PL ML model\ML_predictions.xlsx', index=False)
