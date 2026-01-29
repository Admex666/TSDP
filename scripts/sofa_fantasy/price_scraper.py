import sys
import os
import json
import time
import csv

# Add modules directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.join(current_dir, '..', '..', 'modules')
sys.path.append(modules_dir)

try:
    from SofaScore_module import scrape_sofascore
    print("Imported SofaScore_module successfully.")
except ImportError as e:
    print(f"Error importing SofaScore_module: {e}")
    sys.exit(1)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'player_prices.csv')

# API Endpoint
# https://www.sofascore.com/api/v1/fantasy/round/813/players
BASE_URL = "https://www.sofascore.com/api/v1/fantasy/round/813/players"
RESULTS_PER_PAGE = 50
MAX_PAGES = 100 # Safety limit, will break if no players returned

def scrape_all_prices():
    print(f"Starting Price Scraping from {BASE_URL}...")
    
    all_players = []
    
    for page in range(MAX_PAGES):
        print(f"--- Fetching Page {page} ---")
        
        url = f"{BASE_URL}?page={page}&resultsPerPage={RESULTS_PER_PAGE}&sortParam=form&sortOrder=DESC"
        
        data = scrape_sofascore(url)
        
        if not data or 'players' not in data:
            print("No more data or failed fetch. Stopping.")
            break
            
        players_batch = data['players']
        count = len(players_batch)
        print(f"  Got {count} players.")
        
        if count == 0:
            print("Empty page. Finished.")
            break
            
        for p in players_batch:
            # Extract fields
            # Structure based on test.json:
            # p -> fantasyPlayer -> player -> name
            
            fantasy_player_node = p.get('fantasyPlayer', {})
            player_node_inner = fantasy_player_node.get('player', {})
            
            name = player_node_inner.get('name')
            if not name:
                # Fallback
                name = fantasy_player_node.get('name')
            if not name:
                name = p.get('name', 'Unknown')
                
            slug = player_node_inner.get('slug')
            if not slug:
                slug = fantasy_player_node.get('slug')
            if not slug:
                slug = p.get('slug', '')
                
            # ID: Prefer player_node_inner.id (e.g. 829932), fallback to others
            pid = player_node_inner.get('id')
            if not pid:
                pid = p.get('id')
            
            # Price: p.get('price') seems to mirror fantasyPlayer_node.get('price')
            price = p.get('price')
            if price is None:
                price = fantasy_player_node.get('price', 0)
            
            # Team
            team_info = p.get('team', {})
            if not team_info:
                team_info = fantasy_player_node.get('team', {})
            team_name = team_info.get('name', 'Unknown')
            
            all_players.append({
                'id': pid,
                'name': name,
                'team': team_name,
                'price': price,
                'slug': slug
            })
            
        # Stop early if batch is smaller than requested (last page)
        if count < RESULTS_PER_PAGE:
            print("Last page reached.")
            break
            
    print(f"\nTotal Players Collected: {len(all_players)}")
    
    if all_players:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'name', 'team', 'price', 'slug'])
            writer.writeheader()
            writer.writerows(all_players)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("No players found to save.")

if __name__ == "__main__":
    scrape_all_prices()
