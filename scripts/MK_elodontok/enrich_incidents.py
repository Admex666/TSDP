import sys
import os
import pandas as pd
from tqdm import tqdm

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.SofaScore_module import scrape_sofascore

def enrich_incidents():
    incidents_file = "magyar_kupa_incidents.csv"
    if not os.path.exists(incidents_file):
        print("Incidents file not found.")
        return

    df = pd.read_csv(incidents_file)
    event_ids = df['event_id'].unique()
    
    print(f"Enriching {len(event_ids)} matches with addedTime...")
    
    all_incidents = []
    for event_id in tqdm(event_ids):
        url = f"https://www.sofascore.com/api/v1/event/{event_id}/incidents"
        data = scrape_sofascore(url)
        
        if data and 'incidents' in data:
            for inc in data['incidents']:
                inc_record = {
                    'event_id': event_id,
                    'time': inc.get('time'),
                    'added_time': inc.get('addedTime'),
                    'type': inc.get('incidentType'),
                    'class': inc.get('incidentClass'),
                    'is_home': inc.get('isHome'),
                    'player_name': inc.get('player', {}).get('name') if inc.get('player') else None,
                    'player_id': inc.get('player', {}).get('id') if inc.get('player') else None,
                    'home_score': inc.get('homeScore'),
                    'away_score': inc.get('awayScore'),
                    'description': inc.get('description')
                }
                all_incidents.append(inc_record)
        
    df_new = pd.DataFrame(all_incidents)
    df_new.to_csv(incidents_file, index=False, encoding='utf-8-sig')
    print("Enrichment complete.")

if __name__ == "__main__":
    enrich_incidents()
