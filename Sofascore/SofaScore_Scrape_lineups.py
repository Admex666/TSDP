import requests
import json
import pandas as pd

# Replace this with your actual API URL
URL = "https://www.sofascore.com/api/v1/event/12437015/lineups"

try:
    # Make a GET request to the API
    lineup_response = requests.get(URL)
    
    # Raise an exception for bad status codes
    lineup_response.raise_for_status()
    
    # Get the response text as a string
    json_string_lineup = lineup_response.text

    data = json.loads(json_string_lineup)
    
    # Function to extract player information
    def extract_player_info(player_data):
        player = player_data['player']
        return {
            'name': player['name'],
            'position': player['position'],
            'jersey_number': player['jerseyNumber'],
            'country': player['country']['name'],
            'height': player.get('height', None),
            'substitute': player_data['substitute'],
            'rating': player_data.get('statistics', {}).get('rating', None)
        }
    
    # Extract home and away team lineups
    home_lineup = [extract_player_info(p) for p in data['home']['players']]
    away_lineup = [extract_player_info(p) for p in data['away']['players']]
    
    # Create DataFrames for home and away teams
    df_home = pd.DataFrame(home_lineup)
    df_away = pd.DataFrame(away_lineup)
    
    # Add a column to identify the team
    df_home['team'] = 'Home'
    df_away['team'] = 'Away'
    
    # Combine the DataFrames
    df_lineup = pd.concat([df_home, df_away], ignore_index=True)
    
    # Sort the DataFrame by team and substitute status
    df_lineup = df_lineup.sort_values(['team', 'substitute', 'position'])
    
    # Reset the index
    df_lineup = df_lineup.reset_index(drop=True)
    
    # Display the first few rows of the DataFrame
    print(df_lineup.head())
    
except requests.RequestException as e:
    print(f"HTTP Request failed: {e}")
except json.JSONDecodeError as e:
    print(f"JSON decode error: {e}")