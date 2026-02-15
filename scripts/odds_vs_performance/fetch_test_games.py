import sys
import os
import json
import pandas as pd

# Setup path to import from modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from modules.SofaScore_module import *
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

round_nr = 1
data = scrape_sofascore(f"https://www.sofascore.com/api/v1/unique-tournament/187/season/61714/events/round/{round_nr}")
for event in data["events"]:
    print(event['id'])

print(create_odds_df(event['id']))