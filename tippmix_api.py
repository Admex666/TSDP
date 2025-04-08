import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

url = 'https://api.tippmix.hu/event'
response = requests.get(url)

if response.status_code != 200:
    print(f"Hiba történt: {response.status_code}")


data = response.json()
matches = data['data']

#%%
today_date = datetime.today().date()
span = timedelta(days=200)

# Filter matches
leagues_dict = {'ESP1': 1596, 'ENG1': 1486, 'POR1': 1462, 'ITA1': 1385,
                'FRA1': 951, 'GER1': 517, 'NED1': 1040, 'BEL1': 425}
leagues_dict_inv = {v:k for k, v in leagues_dict.items()}

matches_f = []
for match in matches:
    match_date_iso = datetime.fromisoformat(match['eventDate'])
    match_date = match_date_iso.date()
    filter_ = (
        (match['sportId'] == 1) and
        ( match['competitionId'] in (list(leagues_dict.values())) ) and 
        (match_date <= today_date+span)
        )
    if filter_:
        matches_f.append(match)

# Find odds of filtered matches
odds = []
for match_f in matches_f:
    match_odds = {}
    match_date_iso = datetime.fromisoformat(match_f['eventDate'])
    match_odds['Date'] = datetime.combine(match_date_iso.date(), match_date_iso.time())
    match_odds['HomeTeam'] = match_f['eventParticipants'][0]['participantName']
    match_odds['AwayTeam'] = match_f['eventParticipants'][1]['participantName']
    match_odds['league_name'] = leagues_dict_inv[match_f['competitionId']]
    
    found_1X2, found_Double = False, False
    for market in match_f['markets']:
        # Get 1X2 odds
        if market['marketName'] == '1X2':
            match_odds['1X2_1'] = market['outcomes'][0]['fixedOdds']
            match_odds['1X2_X'] = market['outcomes'][1]['fixedOdds']
            match_odds['1X2_2'] = market['outcomes'][2]['fixedOdds']
            found_1X2 = True
        else:
            pass

        # Get Double chance odds
        if market['marketName'] == 'Kétesély':
            match_odds['Double_1X'] = market['outcomes'][0]['fixedOdds']
            match_odds['Double_12'] = market['outcomes'][1]['fixedOdds']
            match_odds['Double_X2'] = market['outcomes'][2]['fixedOdds']
            found_Double = True
        else:
            pass
        
    if found_1X2 or found_Double:
        odds.append(match_odds)

df_odds = pd.DataFrame(odds)

#%% Fuzz_teams fuzzy matching
path_fuzz = 'ML_PL_new/fuzz_teams.xlsx'
fuzz_teams = pd.read_excel(path_fuzz, sheet_name='cities')

from fuzzywuzzy import fuzz
fuzz_teams['Team_tippmix'] = None

for i_t, row_t in df_odds.iterrows():
    team_tippmix1 = row_t['HomeTeam']
    team_tippmix2 = row_t['AwayTeam']
    country_code = row_t['league_name'][:3]
    
    best_ratio1, best_ratio2 = 0, 0
    for i_fuzz, row_fuzz in fuzz_teams.iterrows():
        if row_fuzz['Country'] == country_code:
            fteam1, fteam2, fteam3 = row_fuzz[['Team_fdcouk', 'Team_fbref', 'Team_odds']]
            
            for teamnr in ['1', '2']:
                for teamcompnr in ['1', '2', '3']:
                    globals()[f'r{teamnr}{teamcompnr}'] = fuzz.ratio(globals()[f'team_tippmix{teamnr}'], globals()[f'fteam{teamcompnr}']) if pd.isna(globals()[f'fteam{teamcompnr}']) == False else 0 
                    
                    #r11 = fuzz.ratio(team_tippmix1, fteam1) if fteam1 != None else 0
                    #r12 = fuzz.ratio(team_tippmix1, fteam2) if fteam2 != None else 0
                    #r13 = fuzz.ratio(team_tippmix1, fteam3) if fteam3 != None else 0
                ratios = [r11, r12, r13] if teamnr == '1' else [r21, r22, r23]
                globals()[f'ratio{teamnr}'] = np.array(ratios).mean()
                if globals()[f'ratio{teamnr}'] > globals()[f'best_ratio{teamnr}']:
                    globals()[f'best_ratio{teamnr}'] = globals()[f'ratio{teamnr}']
                    globals()[f'best_index{teamnr}'] = i_fuzz
    
    fuzz_teams.loc[best_index1, 'Team_tippmix'] = team_tippmix1
    fuzz_teams.loc[best_index2, 'Team_tippmix'] = team_tippmix2
    
#%% Modify 
with pd.ExcelWriter(path_fuzz) as writer:
    fuzz_teams.to_excel(writer, sheet_name='cities', index=False)