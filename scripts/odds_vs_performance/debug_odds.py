import sys
import os
import json

# Setup path to import from modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.SofaScore_module import scrape_sofascore

event_id = 12448542 # Zalaegerszegi TE vs MTK Budapest
url = f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all"
odds_data = scrape_sofascore(url)

if "markets" in odds_data:
    for market in odds_data["markets"]:
        print(f"Market Name: {market.get('marketName')}, Group: {market.get('marketGroup')}")
        if market.get('marketName') == 'Full time' or market.get('marketGroup') == '1X2':
            for choice in market['choices']:
                print(f"  Choice: {choice['name']}, Odds: {choice.get('fractionalValue')}")
else:
    print("No markets found")
