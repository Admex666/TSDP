# collect_snapshots.py

"""
Lépés 2: Minden meccsből snapshots készítése play-by-play alapján
Kimenet: snapshots.csv - minden snapshot egy sor a ML modellhez
"""

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3, boxscoretraditionalv3
import time
import os

def get_game_snapshots(game_id):
    """Egy meccs összes snapshot-ját előállítja"""
    
    # Play-by-play lekérése
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, start_period='1', end_period='4')
    data = pbp.get_dict()
    actions = data['game']['actions']

    # Snapshots: minden negyed vége + 5, 3, 1 perc a végén
    snapshot_times = [
        (1, 360), (1, 0), (2, 360), (2, 0), # Negyed felek 
        (3, 360), (3, 280), (3, 120), (3, 0), # 3. negyed: 6, 4, 2 perc, vége  
        (4, 300), (4, 180), (4, 60), (4, 30)  # 4. negyed: 5, 3, 1, 0.5 perc
    ]
    
    snapshots = []
    
    for period, time_remaining in snapshot_times:
        # Keressük meg a legközelebbi action-t
        snapshot = find_closest_action(actions, period, time_remaining)
        if snapshot:
            snapshot['game_id'] = game_id
            snapshots.append(snapshot)
    
    return snapshots

def find_closest_action(actions, period, target_time):
    """Megkeresi a legközelebbi action-t az adott időpontban"""
    
    best_action = None
    min_diff = float('inf')
    
    for action in actions:
        if action.get('period') != period:
            continue
        
        # Csak olyan action-öket veszünk figyelembe, amiknek van score-ja
        if not action.get('scoreHome') or not action.get('scoreAway'):
            continue
        
        # Idő parse (MM:SS formátum)
        clock = action.get('clock', 'PT00M00S')
        time_sec = parse_clock(clock)
        
        diff = abs(time_sec - target_time)
        if diff < min_diff:
            min_diff = diff
            best_action = action
    
    if not best_action:
        return None
        
    # Snapshot adatok kinyerése
    return {
        'time_remaining_total': 4*720 - (period-1)*720 - (720-target_time),
        'period': period,
        'time_remaining_period': target_time,
        'home_score': int(best_action.get('scoreHome', 0)) if best_action.get('scoreHome') else 0,
        'away_score': int(best_action.get('scoreAway', 0)) if best_action.get('scoreAway') else 0,
        'score_diff': (int(best_action.get('scoreHome', 0)) if best_action.get('scoreHome') else 0) - (int(best_action.get('scoreAway', 0)) if best_action.get('scoreAway') else 0),
        'clock': best_action.get('clock', ''),
        'action_number': best_action.get('actionNumber', 0)
    }

def parse_clock(clock_str):
    """PT12M34S -> 754 másodperc"""
    if not clock_str or clock_str == '':
        return 0
    
    clock_str = clock_str.replace('PT', '').replace('S', '')
    parts = clock_str.split('M')
    
    if len(parts) == 2:
        minutes = int(parts[0]) if parts[0] else 0
        seconds = int(float(parts[1])) if parts[1] else 0
        return minutes * 60 + seconds
    return 0

def process_games(games_csv, output_csv='data/snapshots.csv'):
    """Összes meccs feldolgozása"""
    
    # Már létező snapshots
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        processed_ids = set(existing['game_id'].astype(str))
        print(f"Már feldolgozott meccsek: {len(processed_ids)}")
    else:
        existing = pd.DataFrame()
        processed_ids = set()
    
    # Meccsek betöltése
    games = pd.read_csv(games_csv)
    games['GAME_ID'] = games['GAME_ID'].astype(str)
    
    all_snapshots = []
    
    for idx, game_id in enumerate(games['GAME_ID'].unique()):
        if game_id in processed_ids:
            continue
        
        print(f"[{idx+1}/{len(games['GAME_ID'].unique())}] Feldolgozás: {game_id}")
        
        try:
            snapshots = get_game_snapshots(f"00{game_id}")
            all_snapshots.extend(snapshots)
            
            # Minden meccs után mentés
            if len(all_snapshots) > 0:
                new_df = pd.DataFrame(all_snapshots)
                if len(existing) > 0:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df
                
                combined.to_csv(output_csv, index=False)
                existing = combined
                all_snapshots = []
                processed_ids.add(game_id)
            
            time.sleep(0.6)  # Rate limit
            
        except Exception as e:
            print(f"Hiba {game_id}: {e}")
            time.sleep(2)
            continue
    
    print(f"Kész! Összesen {len(existing)} snapshot")

if __name__ == "__main__":
    process_games('data/games_2024_25.csv', 'data/snapshots_2024_25.csv')