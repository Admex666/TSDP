import pandas as pd
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from scraping.sofascore import create_lineups_df, scrape_sofascore

def debug_stats():
    # Man City vs Burnley (Finished)
    # Use the ID from the earlier test or fetch one. 
    # In test_sofascore.py we found one. Let's find one.
    # 23/24 PL Round 1: Burnley vs Man City.
    # ID: 11352303 (Sample form typical PL IDs)
    # Let's use get_events_for_round to find it again to be sure.
    
    # Use imported function directly
    from scraping.sofascore import get_events_for_round
    events = get_events_for_round(17, 52186, 1)
    target = None
    for e in events:
        if "City" in e['homeTeam']['name'] or "City" in e['awayTeam']['name']:
             if e['status']['type'] == 'finished':
                 target = e
                 break
    
    if not target:
        print("Could not find a match to debug.")
        return

    event_id = target['id']
    print(f"Debugging Event: {target['slug']} ({event_id})")
    
    # 1. Inspect Raw JSON for 'lineups'
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/lineups"
    data = scrape_sofascore(url)
    
    has_stats = False
    if 'home' in data and 'players' in data['home']:
        p1 = data['home']['players'][0]
        print("Sample Player Keys:", p1.keys())
        if 'statistics' in p1:
            print("Statistics found in JSON:", p1['statistics'].keys())
            has_stats = True
        else:
            print("NO 'statistics' key in player object!")
            
    # 2. Inspect DataFrame
    df = create_lineups_df(event_id)
    print("DataFrame Columns:", df.columns.tolist())
    
    if has_stats:
        # Check if stats keys are in columns
        # Print keys only to avoid encoding errors and check content
        row = df.iloc[0].to_dict()
        print("Sample row keys:", list(row.keys()))
        print("Sample stats presence:", 'rating' in row, 'goals' in row, 'totalTackle' in row)
        
        # Try printing repr which is safe
        # print("Sample row (repr):", repr(row))

if __name__ == "__main__":
    debug_stats()
