
import sys
import os
import pandas as pd
import json

# Add the modules directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'modules')))

from SofaScore_module import scrape_sofascore

def get_hungarian_cup_finals():
    tournament_id = 305
    base_api_url = "https://www.sofascore.com/api/v1"
    
    # 1. Get all seasons
    seasons_url = f"{base_api_url}/unique-tournament/{tournament_id}/seasons"
    seasons_data = scrape_sofascore(seasons_url)
    
    if not seasons_data or 'seasons' not in seasons_data:
        print("Could not fetch seasons.", flush=True)
        return []

    seasons = seasons_data['seasons']
    print(f"Found {len(seasons)} seasons.", flush=True)
    
    finals_data = []
    
    for season in seasons:
        season_id = season['id']
        season_name = season['name']
        print(f"Processing season: {season_name} ({season_id})...", flush=True)
        
        # 2. Get rounds for this season
        rounds_url = f"{base_api_url}/unique-tournament/{tournament_id}/season/{season_id}/rounds"
        rounds_data = scrape_sofascore(rounds_url)
        
        if not rounds_data or 'rounds' not in rounds_data:
            print(f"  No rounds found for season {season_name}")
            continue
            
        # 3. Find the 'Final' round
        final_round = None
        for r in rounds_data['rounds']:
            if r.get('slug') == 'final' or 'Döntő' in r.get('name', '') or r.get('name') == 'Final':
                final_round = r
                break
        
        if not final_round:
            # Fallback: check if there's only one round or search for the highest round number if slug isn't 'final'
            # But usually 'final' slug works.
            print(f"  No 'Final' round slug found for season {season_name}")
            continue
            
        round_id = final_round['round']
        print(f"  Final round ID: {round_id}")
        
        # 4. Get events for this round
        # Endpoint: unique-tournament/{tournamentId}/season/{seasonId}/events/round/{roundId}/slug/{roundSlug}
        events_url = f"{base_api_url}/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_id}/slug/final"
        events_data = scrape_sofascore(events_url)
        
        if not events_data or 'events' not in events_data:
            print(f"  No events found for final round of {season_name}")
            continue
            
        for event in events_data['events']:
            match_id = event['id']
            match_slug = event['slug']
            match_url = f"https://www.sofascore.com/football/match/{match_slug}/{match_id}"
            
            finals_data.append({
                'season_name': season_name,
                'season_id': season_id,
                'match_name': event.get('customId', f"{event['homeTeam']['name']} - {event['awayTeam']['name']}"),
                'match_id': match_id,
                'match_url': match_url,
                'winner': event.get('winnerCode'), # 1 for home, 2 for away, 3 for draw
                'home_score': event['homeScore'].get('display'),
                'away_score': event['awayScore'].get('display')
            })
            print(f"  Found final: {match_url}", flush=True)
            
    return finals_data

if __name__ == "__main__":
    finals = get_hungarian_cup_finals()
    
    if finals:
        df = pd.DataFrame(finals)
        df.to_csv('magyar_kupa_finals.csv', index=False)
        print(f"\nSuccessfully saved {len(finals)} final match URLs to magyar_kupa_finals.csv")
    else:
        print("No finals found.")
