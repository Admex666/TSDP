import pandas as pd
from fbref import fbref_module as fbref
from datetime import datetime

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
    
    
    df_fixtures[df_fixtures.Date == df_preds.loc[0, 'Date'].normalize()]
