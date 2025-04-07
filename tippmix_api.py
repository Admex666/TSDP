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
span = timedelta(days=8)

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
