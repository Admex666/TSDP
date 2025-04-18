#pip install understatapi
import pandas as pd
import understatapi

client = understatapi.UnderstatClient()
# Set season, player and league
season = '2024'
player_id = '1119'
league = 'Serie A'

#%% Get shot data of a player
shot_data = pd.DataFrame(client.player(player_id).get_shot_data())

match_ids_season = pd.DataFrame(client.league(league).get_match_data(season))['id']
shot_data_season = shot_data[shot_data['match_id'].astype(str).isin(match_ids_season.values)]

#%% To csv
shot_data_season.to_csv('shot_data.csv', index=False)