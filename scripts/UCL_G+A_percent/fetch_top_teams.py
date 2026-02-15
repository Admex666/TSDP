import sys
import os
import json
import pandas as pd

# Setup path to import from modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from modules.SofaScore_module import scrape_sofascore
except ImportError as e:
    print(f"Error importing module: {e}")
    sys.exit(1)

def main():
    url = "https://www.sofascore.com/api/v1/unique-tournament/7/season/76953/top-teams/overall"
    print(f"Fetching team data from: {url}")
    
    data = scrape_sofascore(url)
    
    if data:
        # Save raw data to inspect
        output_path = os.path.join(current_dir, "sofascore_teams_response.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Full JSON response saved to {output_path}")

        if 'topTeams' in data:
            print(f"\nKeys in topTeams: {list(data['topTeams'].keys())}")
            
            # Look for goals scored
            goals_key = None
            for key in data['topTeams'].keys():
                if 'goals' in key.lower() or 'score' in key.lower():
                    goals_key = key
                    break
            
            if goals_key:
                print(f"\nFound goals key: {goals_key}")
                teams_goals = []
                for team_data in data['topTeams'][goals_key]:
                    team_name = team_data.get('team', {}).get('name')
                    goals = team_data.get('statistics', {}).get(goals_key)
                    
                    if team_name and goals is not None:
                        teams_goals.append({'team_name': team_name, 'total_goals': goals})
                
                df_teams = pd.DataFrame(teams_goals)
                print("\n--- Team Goals Data ---")
                print(df_teams.head())
                
                # Check consistency with top_players.csv if it exists
                csv_path = os.path.join(current_dir, "top_players.csv")
                if os.path.exists(csv_path):
                    df_players = pd.read_csv(csv_path)
                    player_teams = set(df_players['team_name'].unique())
                    api_teams = set(df_teams['team_name'].unique())
                    
                    print(f"\nTeams in player data: {len(player_teams)}")
                    print(f"Teams in API data: {len(api_teams)}")
                    
                    missing_teams = player_teams - api_teams
                    if missing_teams:
                         print(f"Warning: Teams in player data but missing in API team goals data: {missing_teams}")
                    else:
                         print("All teams from player data found in API team goals data.")
                         
                # Save team goals separately
                teams_output_path = os.path.join(current_dir, "team_goals.csv")
                df_teams.to_csv(teams_output_path, index=False)
                print(f"\nTeam goals saved to: {teams_output_path}")

            else:
                print("Could not find a key related to 'goals' or 'score' in topTeams.")
        
    else:
        print("Failed to fetch data (empty response).")

if __name__ == "__main__":
    main()
