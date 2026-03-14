
import sys
import json
import pandas as pd
from tqdm import tqdm
import time
import os

# Add modules directory to path
sys.path.append(r"E:\Data\TSDP\modules")
from SofaScore_module import scrape_sofascore


def main():
    output_file = "all_leagues_matches.csv"
    
    # 1. Load existing data for resuming
    all_matches = []
    scraped_keys = set() # (league_id, round_nr)
    
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            all_matches = df_existing.to_dict('records')
            # Create a set of already scraped (league, round) combinations
            for m in all_matches:
                scraped_keys.add((int(m['league_id']), int(m['round_nr'])))
            print(f"Resuming: found {len(all_matches)} existing matches. Unique rounds: {len(scraped_keys)}")
        except Exception as e:
            print(f"Could not load existing CSV for resume, starting fresh: {e}")

    # 2. Load league IDs
    try:
        with open("top40_leagues.json", "r", encoding='utf-8') as f:
            leagues = json.load(f)
    except Exception as e:
        print(f"Error loading leagues file: {e}")
        return

    # Success counter
    total_matches_saved = len(all_matches)
    
    # Progress bar for leagues
    for entry in tqdm(leagues, desc="Scraping leagues"):
        league_id = entry['id']
        league_name = entry['league']
        
        try:
            # Fetch Seasons
            url_s = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/seasons"
            seasons_data = scrape_sofascore(url_s)
            
            if not seasons_data or 'seasons' not in seasons_data:
                print(f"  [Error] No seasons found for {league_name} ({league_id})")
                continue
                
            if len(seasons_data['seasons']) > 1:
                season_id = seasons_data['seasons'][1]['id']
            else:
                season_id = seasons_data['seasons'][0]['id']
            
            # Fetch Rounds List
            url_r = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/rounds"
            rounds_data = scrape_sofascore(url_r)
            
            if not rounds_data or 'rounds' not in rounds_data:
                print(f"  [Error] No rounds found for {league_name} Season {season_id}")
                continue
            
            # Nested loop for EACH round
            for r_entry in tqdm(rounds_data['rounds'], desc=f"  Rounds of {league_name}", leave=False):
                round_nr = r_entry['round']
                
                # RESUME LOGIC: Check if this league and round combination is already in the CSV
                if (int(league_id), int(round_nr)) in scraped_keys:
                    continue
                
                # Fetch Events for THIS specific round
                url_e = f"https://www.sofascore.com/api/v1/unique-tournament/{league_id}/season/{season_id}/events/round/{round_nr}"
                events_data = scrape_sofascore(url_e)
                
                if not events_data or 'events' not in events_data:
                    continue
                
                round_matches = []
                for event in events_data['events']:
                    if event.get('status', {}).get('type') != 'finished':
                        continue
                    
                    match_info = {
                        'league_name': league_name,
                        'league_id': league_id,
                        'season_id': season_id,
                        'round_nr': round_nr,
                        'match_id': event.get('id'),
                        'start_timestamp': event.get('startTimestamp'),
                        'home_team_name': event.get('homeTeam', {}).get('name'),
                        'home_team_id': event.get('homeTeam', {}).get('id'),
                        'home_score': event.get('homeScore', {}).get('display'),
                        'away_team_name': event.get('awayTeam', {}).get('name'),
                        'away_team_id': event.get('awayTeam', {}).get('id'),
                        'away_score': event.get('awayScore', {}).get('display'),
                        'status': event.get('status', {}).get('type')
                    }
                    round_matches.append(match_info)
                    all_matches.append(match_info)

                # Incremental save after EACH ROUND
                if round_matches:
                    total_matches_saved += len(round_matches)
                    df_incremental = pd.DataFrame(all_matches)
                    df_incremental.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                # Mark as scraped to avoid duplicates in same session if logic changes
                scraped_keys.add((int(league_id), int(round_nr)))
                
        except Exception as e:
            print(f"  [Error] Unexpected error for {league_name}: {e}")
            continue

    if total_matches_saved > 0:
        print(f"\nSuccessfully finished! Total matches in CSV: {total_matches_saved}")
    else:
        print("\nNo finished matches found.")

if __name__ == "__main__":
    main()
