import pandas as pd
from fbref import fbref_module as fbref
from datetime import datetime

fuzz_teams = pd.read_excel('ML_PL_new/fuzz_teams.xlsx')

preds_path = r'C:\Users\Ádám\Dropbox\TSDP_output\PL ML model\ML_predictions.xlsx'
preds_sheets = ['predictions', 'pred_probabilities']

df_preds = pd.read_excel(preds_path, sheet_name=preds_sheets[0])
df_predprobs = pd.read_excel(preds_path, sheet_name=preds_sheets[1])

for countrycode in ['ENG', 'ESP', 'GER', 'ITA', 'FRA']:
    comp_id, league = fbref.team_dict_get(countrycode)
    url_fixtures = f'https://fbref.com/en/comps/{comp_id}/schedule/{league}-Scores-and-Fixtures'
    df_fixtures = fbref.scrape(url_fixtures, f'sched_2024-2025_{comp_id}_1')
    # Set datetime format
    df_fixtures.drop(index=df_fixtures[df_fixtures.Wk=='Wk'].index, inplace=True)
    df_fixtures.Date = pd.to_datetime(df_fixtures.Date)
    # only played games
    #df_fixtures = df_fixtures[df_fixtures.Date <= datetime.today()]
    
    df_preds[['HomeGoals', 'AwayGoals']] = None
    for i, fdcouk_home in enumerate(df_preds.HomeTeam):
        fbref_home = fuzz_teams.loc[fuzz_teams.Team_fdcouk == fdcouk_home,'Team_fbref'].iloc[0]
        mask = (df_fixtures.Date == df_preds.loc[i, 'Date'].normalize()) & (df_fixtures.Home == fbref_home)
        score = df_fixtures.loc[mask, 'Score'].iloc[0]

        if df_fixtures.loc[mask, 'Score'].isna().iloc[0]:
            goals_home, goals_away = None, None
        else:
            goals_home, goals_away = score.split('–')
        
        df_preds.loc[i,['HomeGoals', 'AwayGoals']] = goals_home, goals_away
        
