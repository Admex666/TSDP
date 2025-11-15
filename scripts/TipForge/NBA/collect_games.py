# collect_games.py

"""
Lépés 1: Összes meccs alapadatainak gyűjtése
Kimenet: games.csv - minden meccs game_id-ja és alapinfói
"""

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
import os

def collect_games(season='2024-25', output_file='games.csv'):
    # Ha létezik a fájl, betöltjük
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing_ids = set(existing['game_id'].astype(str))
        print(f"Már létező meccsek: {len(existing_ids)}")
    else:
        existing = pd.DataFrame()
        existing_ids = set()
    
    # API hívás
    print(f"Meccsek lekérése: {season}")
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
    games = log.get_dict()
    
    headers = games['resultSets'][0]['headers']
    rows = games['resultSets'][0]['rowSet']
    
    # DataFrame készítése
    df = pd.DataFrame(rows, columns=headers)
    
    # Game ID normalizálás (egy meccsnek 2 sora van, home és away)
    df['GAME_ID'] = df['GAME_ID'].astype(str)
    unique_games = df.groupby('GAME_ID').first().reset_index()
    
    # Csak új meccsek
    new_games = unique_games[~unique_games['GAME_ID'].isin(existing_ids)]
    
    if len(new_games) > 0:
        # Összefűzés és mentés
        if len(existing) > 0:
            combined = pd.concat([existing, new_games], ignore_index=True)
        else:
            combined = new_games
        
        combined.to_csv(output_file, index=False)
        print(f"Új meccsek: {len(new_games)}, Összes: {len(combined)}")
    else:
        print("Nincs új meccs")
    
    return df

if __name__ == "__main__":
    # 2024-25 szezon
    collect_games('2024-25', 'data/games_2024_25.csv')
    time.sleep(1)
    
    # 2023-24 szezon
    collect_games('2023-24', 'data/games_2023_24.csv')