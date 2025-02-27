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
from fbref import fbref_module as fbref
from ML_PL_new import ML_PL_transform_data as mlpl 

# Loading data from website
url_train = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df_tr = pd.read_csv(url_train)
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df_tr = df_tr[needed_cols]
# create BTTS and O2,5 labels
df_tr['BTTS'] = np.where((df_tr.FTHG!=0)&(df_tr.FTAG!=0),'Yes','No')
df_tr['O/U2.5'] = np.where(df_tr.FTHG+df_tr.FTAG>2.5,'Over','Under')
model_input = mlpl.df_to_model_input(df_tr)

#%% Getting the fresh data for predictions
# football-data.co.uk historical data urls:
csv_name_dict = {'ENG':'E0',
                 'ESP':'SP1', 
                 'GER': 'D1', 
                 'ITA': 'I1',
                 'FRA': 'F1'}

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

predictions_merged = pd.DataFrame()
predicition_probs_merged = pd.DataFrame()
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
    df_week['DateTime'] = df_week.Date + ' ' + df_week.Time.str.split(' ').str.get(0)
    df_week['DateTime'] = pd.to_datetime(df_week.DateTime, format='%Y-%m-%d %H:%M')

    # Fuzzy matched squads list:
    fuzz_teams_all = pd.read_excel('ML_PL_new/fuzz_teams.xlsx')
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
        df_current.loc[x, 'Date'] = df_week.loc[x, 'DateTime']
    
    df_all = pd.concat([df_pred,df_current], ignore_index=True)
    
    model_input_pred = mlpl.df_to_model_input(df_all)
    model_input_pred = model_input_pred.iloc[-nr_matches:,:].reset_index(drop=True)
    
    # Build model
    predictions = model_input_pred[['Date','HomeTeam', 'AwayTeam']].copy()
    predictions['Country'] = countrycode
    prediction_probs = predictions.copy()
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
                prediction_probs[f'{clss}_{m_short}_prob'] = proba[:, i]
        
    predictions_merged = pd.concat([predictions_merged, predictions])
    predicition_probs_merged = pd.concat([predicition_probs_merged, prediction_probs])
    
predictions_merged = predictions_merged.sort_values(by='Date').reset_index(drop=True)
predicition_probs_merged = predicition_probs_merged.sort_values(by='Date').reset_index(drop=True)

#%% Scrape and add odds
#path_odds = r'C:\Users\Adam\.Data files\TSDP\ML_PL_new\modinput_odds.xlsx' 
df_odds_all = pd.read_excel('ML_PL_new/modinput_odds.xlsx') # just set working directory right

#%% To excel
output_path = r'C:\Users\Ádám\Dropbox\TSDP_output\PL ML model\ML_predictions.xlsx'
output_sheets = ['predictions', 'pred_probabilities']
# Read the file first
xlsx_preds = pd.read_excel(output_path, sheet_name=output_sheets[0])
xlsx_predprobs = pd.read_excel(output_path, sheet_name=output_sheets[1])
#Add new data
xlsx_preds_new = pd.concat([xlsx_preds, predictions_merged]).sort_values(by='Date').reset_index(drop=True)
xlsx_predprobs_new = pd.concat([xlsx_predprobs, predicition_probs_merged]).sort_values(by='Date').reset_index(drop=True)
#Drop duplicates
xlsx_preds_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
xlsx_predprobs_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)

# Modify excel file
with pd.ExcelWriter(output_path) as writer:
    predictions_merged.to_excel(writer, sheet_name=output_sheets[0], index=False)
    predicition_probs_merged.to_excel(writer, sheet_name=output_sheets[1], index=False)
