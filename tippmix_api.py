import requests
import json
import pandas as pd
from datetime import datetime, timedelta

url = 'https://api.tippmix.hu/event'
response = requests.get(url)

if response.status_code != 200:
    print(f"Hiba történt: {response.status_code}")


data = response.json()
matches = data['data']

#%%
today_date = datetime.today().date()
span = timedelta(days=4)

matches_f = []
for match in matches:
    match_date_iso = datetime.fromisoformat(match['eventDate'])
    match_date = match_date_iso.date()
    filter_ = (match['sportId'] == 1) & ('Augsburg' in match['eventName']) & (match_date <= today_date+span)
    if filter_:
        matches_f.append(match)

for match_f in matches_f:
    for market in match_f['markets']:
        if market['marketName'] == '1X2':
            for outcome in market['outcomes']:
                print(outcome['outcomeName'], outcome['fixedOdds'])
    print("")
