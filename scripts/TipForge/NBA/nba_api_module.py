# Season game log
from nba_api.stats.endpoints import leaguegamelog
def get_season_game_log(season='2025-26'):
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
    games = log.get_dict()

    print(f"Total games found: {len(games['resultSets'][0]['rowSet'])}")
    
    headers = games['resultSets'][0]['headers']
    rows = games['resultSets'][0]['rowSet']
    
    # DataFrame készítése
    df = pd.DataFrame(rows, columns=headers)
    
    # Game ID normalizálás (egy meccsnek 2 sora van, home és away)
    df['GAME_ID'] = df['GAME_ID'].astype(str)

    return df


# Play by play
from nba_api.stats.endpoints import playbyplayv3
import pandas as pd
def get_game_snapshots(game_id):
    plays_log = playbyplayv3.PlayByPlayV3(game_id=game_id, 
                            start_period='1', 
                            end_period='4')
    plays = plays_log.get_dict()

    actions = plays['game']['actions']
    #for act in actions:
    #    print(act['actionType'], act['clock'])
    #print(len(actions))

    return pd.DataFrame(actions)