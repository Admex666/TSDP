import sys
import os
import json
import pandas as pd

# Setup path to import from modules
# We are in c:\Users\Adam\Data\TSDP\scripts\UCL_G+A_percent
# We need to reach c:\Users\Adam\Data\TSDP\modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from modules.SofaScore_module import scrape_sofascore
except ImportError as e:
    print(f"Error importing module: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def main():
    url = "https://www.sofascore.com/api/v1/unique-tournament/7/season/76953/top-players/overall"
    print(f"Fetching data from: {url}")
    
    data = scrape_sofascore(url)
    
    if data:
        if 'topPlayers' in data:
            all_players_dict = {}
            
            # Iterate over all categories in topPlayers (rating, goals, assists, etc.)
            for category, players_list in data['topPlayers'].items():
                print(f"Processing category: {category} ({len(players_list)} items)")
                
                for p in players_list:
                    player_info = p.get('player', {})
                    player_id = player_info.get('id')
                    
                    if not player_id:
                        continue
                        
                    # Initialize player entry if not exists
                    if player_id not in all_players_dict:
                        team_info = p.get('team', {})
                        all_players_dict[player_id] = {
                            # Player Info
                            'player_name': player_info.get('name'),
                            'player_id': player_id,
                            'position': player_info.get('position'),
                            'player_slug': player_info.get('slug'),
                            
                            # Team Info
                            'team_name': team_info.get('name'),
                            'team_id': team_info.get('id'),
                            'team_slug': team_info.get('slug'),
                        }
                    
                    # Merge statistics
                    stats_info = p.get('statistics', {})
                    for stat_key, stat_value in stats_info.items():
                        if stat_key != 'id': # Skip ID in stats to avoid conflict/redundancy
                            all_players_dict[player_id][stat_key] = stat_value

            processed_players = list(all_players_dict.values())
            df = pd.DataFrame(processed_players)
            
            # Display first few rows
            print("\n--- DataFrame Head ---")
            print(df.head())
            print(f"\nTotal unique players found: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            
            # Save to CSV
            output_csv = os.path.join(current_dir, "top_players.csv")
            df.to_csv(output_csv, index=False)
            print(f"\nDataFrame saved to: {output_csv}")
            
        else:
            print("Could not find 'topPlayers.rating' in response.")
            if 'topPlayers' in data:
                 if isinstance(data['topPlayers'], dict):
                     print(f"Keys available in 'topPlayers': {list(data['topPlayers'].keys())}")
                 else:
                     print(f"'topPlayers' is of type: {type(data['topPlayers'])}")
        
        # Save raw data to inspect
        output_path = os.path.join(current_dir, "sofascore_response.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Full JSON response saved to {output_path}")
        
    else:
        print("Failed to fetch data (empty response).")

if __name__ == "__main__":
    main()
