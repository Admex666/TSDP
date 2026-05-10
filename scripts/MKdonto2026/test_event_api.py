
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'modules')))
from SofaScore_module import scrape_sofascore

# Try a match ID
event_id = 16079853
url = f"https://www.sofascore.com/api/v1/event/{event_id}"
print(f"Fetching {url}...")
data = scrape_sofascore(url)
print(f"Data keys: {data.keys() if data else 'None'}")
if data and 'event' in data:
    print(f"Match: {data['event']['homeTeam']['name']} vs {data['event']['awayTeam']['name']}")
