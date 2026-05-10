
import sys
import os
import pandas as pd
import json

# Add the modules directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'modules')))

from SofaScore_module import (
    scrape_sofascore, 
    fetch_match_details, 
    fetch_match_incidents, 
    create_odds_df
)

def process_cup_finals():
    csv_path = 'magyar_kupa_finals.csv'
    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return

    df_finals = pd.read_csv(csv_path)
    
    # Filter for seasons from 09/10 onwards
    # The seasons are named "Magyar Kupa YY/YY"
    # We can filter by checking if it's NOT 08/09 or earlier (if any)
    # Looking at the CSV, we have 08/09 at the end.
    
    # A simple way: find the index of the first '08/09' and take everything before it.
    # Or just check the year.
    def is_target_season(season_name):
        if not isinstance(season_name, str): return False
        if '08/09' in season_name: return False
        if '07/08' in season_name or '06/07' in season_name: return False # Just in case
        return True

    df_filtered = df_finals[df_finals['season_name'].apply(is_target_season)].copy()
    print(f"Processing {len(df_filtered)} matches...")

    all_match_details = []
    all_incidents = []
    all_odds = []

    for index, row in df_filtered.iterrows():
        match_id = row['match_id']
        season = row['season_name']
        match_url = row['match_url']
        print(f"Fetching details for {season}: {match_url} (ID: {match_id})...")
        
        # 1. Match Details
        details = fetch_match_details(match_id, referer=match_url)
        if details:
            details['season'] = season
            all_match_details.append(details)
        
        # 2. Incidents
        incidents_df = fetch_match_incidents(match_id, referer=match_url)
        if not incidents_df.empty:
            incidents_df['match_id'] = match_id
            incidents_df['season'] = season
            all_incidents.append(incidents_df)
            
        # 3. Odds
        odds_df = create_odds_df(match_id)
        if not odds_df.empty:
            odds_df['match_id'] = match_id
            odds_df['season'] = season
            all_odds.append(odds_df)

    # Save results
    if all_match_details:
        pd.DataFrame(all_match_details).to_csv('match_details_09_26.csv', index=False)
        print("Saved match_details_09_26.csv")
        
    if all_incidents:
        pd.concat(all_incidents).to_csv('match_incidents_09_26.csv', index=False)
        print("Saved match_incidents_09_26.csv")
        
    if all_odds:
        pd.concat(all_odds).to_csv('match_odds_09_26.csv', index=False)
        print("Saved match_odds_09_26.csv")

if __name__ == "__main__":
    process_cup_finals()
