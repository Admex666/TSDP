import sys
import os
import pandas as pd
import time
from tqdm import tqdm

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.SofaScore_module import scrape_sofascore, create_odds_df

def save_incremental(data, filename):
    """Saves a single record or list of records to CSV incrementally."""
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')

def run_full_collection():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    matches_file = os.path.join(output_dir, "magyar_kupa_matches_sofascore.csv")
    incidents_file = os.path.join(output_dir, "magyar_kupa_incidents.csv")
    teams_file = os.path.join(output_dir, "magyar_kupa_teams_leagues.csv")

    # Clean existing files if we want a fresh start (optional, but usually safer for a script like this)
    for f in [matches_file, incidents_file, teams_file]:
        if os.path.exists(f):
            os.remove(f)

    # 1. Fetch Cup Tree to get event IDs
    cup_tree_url = "https://www.sofascore.com/api/v1/unique-tournament/305/season/81521/cuptrees"
    print(f"Fetching cup tree: {cup_tree_url}")
    tree_data = scrape_sofascore(cup_tree_url)
    
    if not tree_data or 'cupTrees' not in tree_data:
        print("Failed to fetch or parse cup tree data.")
        return

    event_ids = []
    round_mapping = {}
    
    for tree in tree_data['cupTrees']:
        for round_info in tree.get('rounds', []):
            round_name = round_info.get('description') or round_info.get('type') or "Unknown Round"
            for block in round_info.get('blocks', []):
                for event_id in block.get('events', []):
                    if isinstance(event_id, int):
                        if event_id not in event_ids:
                            event_ids.append(event_id)
                            round_mapping[event_id] = round_name

    print(f"Total events to scrape: {len(event_ids)}")
    
    unique_teams = {} # team_id -> team_name
    processed_teams = set() # To avoid duplicate team scrapes
    
    # 2. Scrape each event
    for event_id in tqdm(event_ids, desc="Scraping matches & incidents"):
        # A. Fetch event details
        event_url = f"https://www.sofascore.com/api/v1/event/{event_id}"
        event_data = scrape_sofascore(event_url)
        
        if not event_data or 'event' not in event_data:
            print(f"\nFailed to fetch event {event_id}")
            continue
            
        event = event_data['event']
        home = event['homeTeam']
        away = event['awayTeam']
        
        # Track unique teams
        unique_teams[home['id']] = home['name']
        unique_teams[away['id']] = away['name']
        
        # Get scores
        home_score = event.get('homeScore', {}).get('display', 0)
        away_score = event.get('awayScore', {}).get('display', 0)
        
        # B. Get odds
        df_odds = create_odds_df(event_id)
        odds_1, odds_X, odds_2 = None, None, None
        if not df_odds.empty:
            for _, row in df_odds.iterrows():
                if row['name'] == '1': odds_1 = round(row['odds'], 2)
                elif row['name'] == 'X': odds_X = round(row['odds'], 2)
                elif row['name'] == '2': odds_2 = round(row['odds'], 2)
        
        match_record = {
            'event_id': event_id,
            'round': round_mapping.get(event_id),
            'home_team': home['name'],
            'home_id': home['id'],
            'away_team': away['name'],
            'away_id': away['id'],
            'score_home': home_score,
            'score_away': away_score,
            'odds_1': odds_1,
            'odds_X': odds_X,
            'odds_2': odds_2,
            'status': event.get('status', {}).get('description')
        }
        save_incremental(match_record, matches_file)

        # C. Fetch incidents (Match sheet)
        incidents_url = f"https://www.sofascore.com/api/v1/event/{event_id}/incidents"
        incidents_data = scrape_sofascore(incidents_url)
        
        if incidents_data and 'incidents' in incidents_data:
            for inc in incidents_data['incidents']:
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
                save_incremental(inc_record, incidents_file)

    # 3. Scrape team tournaments
    print(f"\nFetching tournament info for {len(unique_teams)} teams...")
    for team_id, team_name in tqdm(unique_teams.items(), desc="Scraping teams"):
        if team_id in processed_teams:
            continue
            
        tourney_url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournaments"
        tourney_data = scrape_sofascore(tourney_url)
        
        leagues = []
        if tourney_data:
            if 'uniqueTournaments' in tourney_data:
                for ut in tourney_data['uniqueTournaments']:
                    leagues.append(ut.get('name'))
            elif 'groups' in tourney_data:
                for group in tourney_data['groups']:
                    for ut in group.get('uniqueTournaments', []):
                        leagues.append(ut.get('name'))
        
        team_record = {
            'team_id': team_id,
            'team_name': team_name,
            'leagues': "; ".join(leagues)
        }
        save_incremental(team_record, teams_file)
        processed_teams.add(team_id)

    print(f"\nSikeresen mentve: {matches_file}, {incidents_file}, {teams_file}")

if __name__ == "__main__":
    run_full_collection()
