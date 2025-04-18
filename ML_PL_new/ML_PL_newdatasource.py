import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
wd_old = os.getcwd()
if wd_old != 'C:\\Users\\Adam\\..Data\\TSDP':
    wd_base = wd_old.split('\\')[:4]
    wd_new = '\\'.join(wd_base)+'\\TSDP'
    os.chdir(wd_new)
from fbref import fbref_module as fbref
from ML_PL_new.ML_PL_transform_data import df_to_model_input
import datetime

#%% Loading data from website
url_train = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df_tr = pd.read_csv(url_train)
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df_tr = df_tr[needed_cols]
# create BTTS and O2,5 labels
df_tr['BTTS'] = np.where((df_tr.FTHG!=0)&(df_tr.FTAG!=0),'Yes','No')
df_tr['O/U2.5'] = np.where(df_tr.FTHG+df_tr.FTAG>2.5,'Over','Under')
model_input = df_to_model_input(df_tr, weather=False)

#%% Getting the fresh data for predictions
# football-data.co.uk historical data urls:
csv_name_dict = {'ENG':'E0', 'ESP':'SP1', 'GER': 'D1', 'ITA': 'I1', 'FRA': 'F1',
                 'NED': 'N1', 'BEL': 'B1', 'POR': 'P1'}

params_grid = {'GaussianNB':{},
               'DecisionTreeClassifier': {'max_depth': 4, 'min_samples_split': 6,
                                          'min_samples_leaf': 2, 'criterion':'entropy'
                                          },
               'RandomForestClassifier': {'max_depth': 7, 'min_samples_split': 5,
                                          'min_samples_leaf': 2, 'max_features': 'sqrt', 
                                          'bootstrap': True      
                                          },
               'KNeighborsClassifier': {'algorithm': 'brute',
                                        'leaf_size': 10,
                                        'n_neighbors': 8,
                                        'p': 2,
                                        'weights': 'distance'
                                        }
               }

model_short_dict = {'GaussianNB':'gNB', 
                    'RandomForestClassifier': 'RF',
                    'DecisionTreeClassifier': 'DT',
                    'KNeighborsClassifier': 'KNN'}

model_list = ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier',
              'KNeighborsClassifier']

predictions_merged = pd.DataFrame()
predicition_probs_merged = pd.DataFrame()
fuzz_teams_all = pd.read_excel('ML_PL_new/fuzz_teams.xlsx', sheet_name='cities')

for countrycode in csv_name_dict.keys():
    csv_name = csv_name_dict[countrycode]

    path_pred = f"https://www.football-data.co.uk/mmz4281/2425/{csv_name}.csv"
    df_pred = pd.read_csv(path_pred)
    
    # Transform
    df_pred = df_pred[needed_cols]
    # create BTTS and O2,5 labels
    df_pred['BTTS'] = np.where((df_pred.FTHG!=0)&(df_pred.FTAG!=0),'Yes','No')
    df_pred['O/U2.5'] = np.where(df_pred.FTHG+df_pred.FTAG>2.5,'Over','Under')
    teams = np.sort(df_pred.HomeTeam.dropna().unique())

    # Create next round's pairings
    df_current = df_pred.copy()
    df_current.iloc[:,3:-3] = 0
    df_current[['FTR','BTTS','O/U2.5']] = 'none'
    
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    # Set datetime format
    df_fixtures.drop(index=df_fixtures[df_fixtures.Wk=='Wk'].index, inplace=True)
    df_fixtures.Date = pd.to_datetime(df_fixtures.Date)
    today = datetime.datetime.today()
    span = pd.to_timedelta('8D')
    mask_date = (df_fixtures.Date <= today + span) & (df_fixtures.Date >= today)
    while (True not in mask_date.unique()) and (span < pd.to_timedelta('10D')):
        span += pd.to_timedelta('1D')
        mask_date = (df_fixtures.Date < today + span) & (df_fixtures.Date > today)
    # Find games in n days span
    df_week = df_fixtures.loc[mask_date, :].reset_index(drop=True)
    if df_week.empty:
        print(f"No games found in {span.days} days span in {league}.")
        continue
    else:
        print(f'{len(df_week)} games found in the {league}.')
    df_week['DateTime'] = df_week.Date.astype(str) + ' ' + df_week.Time.str.split(' ').str.get(0)
    df_week['DateTime'] = pd.to_datetime(df_week.DateTime, format='%Y-%m-%d %H:%M')
    nr_matches = len(df_week)
    df_current = df_current.iloc[:nr_matches]
    
    # Fuzzy matched squads list:
    fuzz_teams = fuzz_teams_all[fuzz_teams_all.Country == countrycode]
    
    for fbrteam in [*df_week.Home.unique(), *df_week.Away.unique()]:
        i = fuzz_teams[fuzz_teams.Team_fbref == fbrteam].index[0]
        if fbrteam in df_week['Home'].unique():
            home_away = 'Home'
        elif fbrteam in df_week['Away'].unique():
            home_away = 'Away'
        else:
            home_away = 'ERROR'
        
        df_current[['Date', 'fbrHomeTeam', 'fbrAwayTeam']] = df_week[['DateTime', 'Home', 'Away']]
    
    for x in range(len(df_current)):
        mask_home = fuzz_teams.Team_fbref == df_current.fbrHomeTeam[x]
        mask_away = fuzz_teams.Team_fbref == df_current.fbrAwayTeam[x]
        df_current.loc[x, ['HomeTeam', 'AwayTeam']] = [fuzz_teams.loc[mask_home,'Team_fdcouk'].iloc[0],
                                                       fuzz_teams.loc[mask_away,'Team_fdcouk'].iloc[0]]
    df_current.drop(columns=['fbrHomeTeam', 'fbrAwayTeam'], inplace=True)
    
    df_all = pd.concat([df_pred,df_current], ignore_index=True)
    
    model_input_pred = df_to_model_input(df_all, weather=False)
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
        for m in model_list:
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
    
if predictions_merged.empty == False:
    predictions_merged = predictions_merged.sort_values(by='Date').reset_index(drop=True)
    predicition_probs_merged = predicition_probs_merged.sort_values(by='Date').reset_index(drop=True)

#%% Scrape and add odds
path_odds = r'ML_PL_new\modinput_odds.xlsx' # just set working directory right
df_odds_all = pd.read_excel(path_odds) 
df_wh = df_odds_all.loc[df_odds_all.bookmaker=='William Hill', :].drop(columns='Date').reset_index(drop=True)

for dfname, newdfname in zip(['predicition_probs', 'predictions'], ['predprob', 'pred']):
    # Merge data
    globals()[f'{dfname}_merged'][['Home', 'Away']] = None
    for i in range(len(globals()[f'{dfname}_merged'])):
        fdcouk_home = globals()[f'{dfname}_merged'].loc[i, 'HomeTeam']
        fdcouk_away = globals()[f'{dfname}_merged'].loc[i, 'AwayTeam']
        odds_home = fuzz_teams_all.loc[fuzz_teams_all.Team_fdcouk == fdcouk_home, 'Team_odds'].iloc[0]
        odds_away = fuzz_teams_all.loc[fuzz_teams_all.Team_fdcouk == fdcouk_away, 'Team_odds'].iloc[0]
        globals()[f'{dfname}_merged'].loc[i, ['Home', 'Away']] = odds_home, odds_away
    
    globals()[f'df_{newdfname}_odds'] = pd.merge(globals()[f'{dfname}_merged'], df_wh,
                                             on=['Home', 'Away'])
    globals()[f'df_{newdfname}_odds'].drop(columns=['Home','Away'], inplace=True)

# See value
for out in ['H', 'D', 'A', 'Over', 'Under']:
    for m_short in model_short_dict.values():
        df_predprob_odds[f'{out}_{m_short}_value'] = (1/df_predprob_odds[f'{out}_odds'] / df_predprob_odds[f'{out}_{m_short}_prob'] -1) *100

# Calculate bet size
def calc_bet_size_propo(bankroll, odds_bookie, prob_fair):
    # proportional = ( (myprob-prob) * bankroll / (odds-1))
    prob_bookie = 1/odds_bookie
    bet_size = (prob_fair - prob_bookie) * bankroll / (odds_bookie-1) /7
    return bet_size

for out in ['H', 'D', 'A', 'Over', 'Under']:
    for m_short in model_short_dict.values():
        criterion = (df_predprob_odds[f'{out}_{m_short}_value'] >= 0)
        odds_bookie = df_predprob_odds[f'{out}_odds']
        prob_fair = df_predprob_odds[f'{out}_{m_short}_prob']
        bet_col = calc_bet_size_propo(100/3, odds_bookie, prob_fair)
        df_predprob_odds[f'{out}_{m_short}_bet'] = bet_col
        df_predprob_odds[f'{out}_{m_short}_bet'] = np.where(criterion,
                                                            df_predprob_odds[f'{out}_{m_short}_bet'],
                                                            0)
# Calculate bets for df_pred_odds     
for out in ['H', 'D', 'A']:
    for m_short in model_short_dict.values():
        df_pred_odds[f'{out}_{m_short}_bet'] = np.where(df_pred_odds[f'FTR_{m_short}'] == out,
                                                        1/(df_pred_odds[f'{out}_odds']-1),
                                                        0)
        
for out in ['Over', 'Under']:
    for m_short in model_short_dict.values():
        df_pred_odds[f'{out}_{m_short}_bet'] = np.where(df_pred_odds[f'O/U2.5_{m_short}'] == out,
                                                        1/(df_pred_odds[f'{out}_odds']-1),
                                                        0)

#%% To excel
output_path = 'ML_PL_new/predictions.xlsx'
output_sheets = ['predictions', 'pred_probabilities']
# Read the file first
xlsx_preds = pd.read_excel(output_path, sheet_name=output_sheets[0])
xlsx_predprobs = pd.read_excel(output_path, sheet_name=output_sheets[1])
#Add new data
xlsx_preds_new = pd.concat([xlsx_preds, df_pred_odds]).sort_values(by='Date').reset_index(drop=True)
xlsx_predprobs_new = pd.concat([xlsx_predprobs, df_predprob_odds]).sort_values(by='Date').reset_index(drop=True)
#Drop duplicates
xlsx_preds_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
xlsx_predprobs_new.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True)
print(f'Number of new rows: {len(df_pred_odds)}')

#%% Modify excel file
with pd.ExcelWriter(output_path) as writer:
    xlsx_preds_new.to_excel(writer, sheet_name=output_sheets[0], index=False)
    xlsx_predprobs_new.to_excel(writer, sheet_name=output_sheets[1], index=False)