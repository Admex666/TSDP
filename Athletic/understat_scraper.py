def get_player_shotmap(season, player_id, league):
    #pip install understatapi
    import pandas as pd
    import understatapi
    
    client = understatapi.UnderstatClient()
    
    # Get shot data of a player
    shot_data = pd.DataFrame(client.player(player_id).get_shot_data())
    
    match_ids_season = pd.DataFrame(client.league(league).get_match_data(season))['id']
    shot_data_season = shot_data[shot_data['match_id'].astype(str).isin(match_ids_season.values)].reset_index(drop=True)
    shot_data_season[['X', 'Y', 'xG', 'minute', 'h_goals', 'a_goals']] = shot_data_season[['X', 'Y', 'xG', 'minute', 'h_goals', 'a_goals']].astype(float)
    shot_data_season.date = pd.to_datetime(shot_data_season.date)
    
    return shot_data_season