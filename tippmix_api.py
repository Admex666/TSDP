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
    match_odds['Home'] = match_f['eventParticipants'][0]['participantName']
    match_odds['Away'] = match_f['eventParticipants'][1]['participantName']
    match_odds['league_name'] = leagues_dict_inv[match_f['competitionId']]
    
    found_1X2, found_Double, found_o_u = False, False, False
    for market in match_f['markets']:
        # Get 1X2 odds
        if market['marketName'] == '1X2':
            match_odds['H_odds'] = market['outcomes'][0]['fixedOdds']
            match_odds['D_odds'] = market['outcomes'][1]['fixedOdds']
            match_odds['A_odds'] = market['outcomes'][2]['fixedOdds']
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
        
        # O/U odds
        if market['marketName'] == 'Gólszám 2,5':
            match_odds['Under_odds'] = market['outcomes'][0]['fixedOdds']
            match_odds['Over_odds'] = market['outcomes'][1]['fixedOdds']
            found_o_u = True
        else:
            pass
        
    if found_1X2 or found_Double:
        odds.append(match_odds)

df_odds = pd.DataFrame(odds)

#%% Update excel
path = 'ML_PL_new/modinput_odds_tx.xlsx'
odds_tx_prev = pd.read_excel(path)
odds_tx_combined = pd.concat([odds_tx_prev, df_odds], ignore_index=True)
odds_tx_new = odds_tx_combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'],
                                               keep='first').reset_index(drop=True)

odds_tx_new.to_excel(path, index=False)
print(f'Appended data with {len(odds_tx_new) - len(odds_tx_prev)} games.')

#%% Fuzz_teams fuzzy matching
path_fuzz = 'ML_PL_new/fuzz_teams.xlsx'
fuzz_teams_og = pd.read_excel(path_fuzz, sheet_name='cities')
# Only None values to modify
fuzz_teams = fuzz_teams_og[pd.isna(fuzz_teams_og.Team_tippmix)]

from fuzzywuzzy import fuzz

for i_fuzz, row_fuzz in fuzz_teams.iterrows():
    best_ratio1, best_ratio2 = 0, 0
    for i_t, row_t in df_odds.iterrows():
        team_tippmix1 = row_t['HomeTeam']
        team_tippmix2 = row_t['AwayTeam']
        country_code = row_t['league_name'][:3]
        
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
        else:
            continue
        
        fuzz_teams_og.loc[best_index1, 'Team_tippmix'] = team_tippmix1 if best_ratio1 > 70 else None
        fuzz_teams_og.loc[best_index2, 'Team_tippmix'] = team_tippmix2 if best_ratio2 > 70 else None
    
#%% Modify 
with pd.ExcelWriter(path_fuzz) as writer:
    fuzz_teams_og.to_excel(writer, sheet_name='cities', index=False)