import json
import os
import pandas as pd
from datetime import datetime

def normalize_name(name):
    if not name: return ""
    return "".join(e for e in name.lower() if e.isalnum())

def merge_data():
    results_path = 'data/historical_results_combined.json'
    odds_path = 'data/historical_odds_lovi.json'
    output_path = 'data/training_set_v2_with_odds.csv'

    if not os.path.exists(results_path) or not os.path.exists(odds_path):
        print("Required files missing.")
        return

    print("Loading datasets...")
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    with open(odds_path, 'r', encoding='utf-8') as f:
        odds_data = json.load(f)

    # Build odds lookup
    # Key: (date, normalized_horse_name) -> odds
    odds_lookup = {}
    for day in odds_data.get('days', []):
        date = day['date']
        for race in day.get('races', []):
            for p in race.get('participants', []):
                h_name = normalize_name(p.get('horse', ''))
                if not h_name: continue
                
                # Key: (date, h_name)
                key = (date, h_name)
                odds_val = p.get('starting_odds', '-').replace(',', '.')
                try:
                    # Ha már van ilyen kulcs, ne írjuk felül (ha ugyanaz a ló kétszer futna, 
                    # ami ritka, az elsőt vesszük vagy logolunk)
                    if key not in odds_lookup:
                        odds_lookup[key] = float(odds_val)
                except:
                    if key not in odds_lookup:
                        odds_lookup[key] = None

    print(f"Built odds lookup for {len(odds_lookup)} horses.")

    # Merge into a flat list for CSV
    merged_rows = []
    match_count = 0
    total_participants = 0

    for race in results_data.get('races', []):
        date = race.get('race_date')
        start_time = race.get('start')
        race_name = race.get('race_name')
        
        for p in race.get('participants', []):
            total_participants += 1
            h_name = normalize_name(p.get('name', ''))
            
            key = (date, h_name)
            market_odds = odds_lookup.get(key)
            
            if market_odds is not None:
                match_count += 1
            
            row = {
                "race_id": race.get('race_id'),
                "date": date,
                "start": start_time,
                "race_name": race_name,
                "horse_id": p.get('id'),
                "horse_name": p.get('name'),
                "program_num": p.get('number'),
                "rank": p.get('rank'),
                "market_odds": market_odds,
                "is_win": 1 if p.get('rank') == "1." else 0
            }
            merged_rows.append(row)

    print(f"Match results: {match_count}/{total_participants} ({match_count/total_participants:.1%})")
    
    df = pd.DataFrame(merged_rows)
    df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == "__main__":
    merge_data()
