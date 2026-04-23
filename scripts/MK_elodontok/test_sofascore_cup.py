import sys
import os
import pandas as pd
import json

# Add modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.SofaScore_module import scrape_sofascore, create_odds_df

def test_cup_collection():
    cup_tree_url = "https://www.sofascore.com/api/v1/unique-tournament/305/season/81521/cuptrees"
    print(f"Fetching cup tree: {cup_tree_url}")
    tree_data = scrape_sofascore(cup_tree_url)
    
    if not tree_data or 'cupTrees' not in tree_data:
        print("Failed to fetch or parse cup tree data.")
        return

    # Collect all event IDs first
    event_ids = []
    round_names = {} # event_id -> round_name
    
    for tree in tree_data['cupTrees']:
        for round_info in tree.get('rounds', []):
            round_name = round_info.get('description') or round_info.get('type') or "Unknown Round"
            for block in round_info.get('blocks', []):
                for event_id in block.get('events', []):
                    if isinstance(event_id, int):
                        event_ids.append(event_id)
                        round_names[event_id] = round_name

    print(f"Total events found in tree: {len(event_ids)}")
    
    all_events = []
    # Test only first 5 events for a better overview
    for event_id in event_ids[:5]:
        print(f"\n--- Processing Event {event_id} ---")
        
        # 1. Fetch event details
        event_url = f"https://www.sofascore.com/api/v1/event/{event_id}"
        event_data = scrape_sofascore(event_url)
        
        if not event_data or 'event' not in event_data:
            print(f"Failed to fetch event {event_id}")
            continue
            
        event = event_data['event']
        home_team = event['homeTeam']['name']
        away_team = event['awayTeam']['name']
        home_id = event['homeTeam']['id']
        home_score = event.get('homeScore', {}).get('display', 0)
        away_score = event.get('awayScore', {}).get('display', 0)
        
        # 2. Fetch odds
        print(f"Fetching odds for {home_team} vs {away_team}...")
        df_odds = create_odds_df(event_id)
        
        odds_1, odds_X, odds_2 = None, None, None
        if not df_odds.empty:
            for _, row in df_odds.iterrows():
                if row['name'] == '1': odds_1 = round(row['odds'], 2)
                elif row['name'] == 'X': odds_X = round(row['odds'], 2)
                elif row['name'] == '2': odds_2 = round(row['odds'], 2)

        # 3. Fetch tournaments for home team
        tourney_url = f"https://www.sofascore.com/api/v1/team/{home_id}/unique-tournaments"
        print(f"Fetching tournaments for {home_team}...")
        tourney_data = scrape_sofascore(tourney_url)
        
        league_names = []
        if tourney_data:
            # Check for 'groups' (older API or certain teams)
            if 'groups' in tourney_data:
                for group in tourney_data['groups']:
                    for ut in group.get('uniqueTournaments', []):
                        league_names.append(ut.get('name'))
            # Check for 'uniqueTournaments' directly (newer API)
            elif 'uniqueTournaments' in tourney_data:
                for ut in tourney_data['uniqueTournaments']:
                    league_names.append(ut.get('name'))
        
        all_events.append({
            'Kör': round_names.get(event_id),
            'Hazai': home_team,
            'Vendég': away_team,
            'Eredmény': f"{home_score}:{away_score}",
            'Odds_1': odds_1,
            'Odds_X': odds_X,
            'Odds_2': odds_2,
            'Hazai Ligái': ", ".join(league_names[:2])
        })

    print("\n--- Final Test Results (SofaScore) ---")
    df = pd.DataFrame(all_events)
    # Ensure correct encoding for output
    print(df.to_string())

if __name__ == "__main__":
    test_cup_collection()
