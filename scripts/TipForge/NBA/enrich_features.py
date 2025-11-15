# enrich_features.py

"""
Lépés 3: Snapshots kiegészítése további feature-ökkel
Kimenet: ml_ready.csv - teljes feature set ML-hez
"""

import pandas as pd
from nba_api.stats.endpoints import (
    boxscoretraditionalv3, 
    playbyplayv3,
    teamgamelog,
    leaguedashteamstats
)
import time
import os
from collect_snapshots import parse_clock

def get_team_season_stats(team_id, season='2024-25'):
    """Csapat szezon statisztikák"""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season'
        )
        df = stats.get_data_frames()[0]
        team_stats = df[df['TEAM_ID'] == team_id]
        
        if len(team_stats) == 0:
            return {}
        
        return {
            'off_rating': team_stats['OFF_RATING'].values[0],
            'def_rating': team_stats['DEF_RATING'].values[0],
            'net_rating': team_stats['NET_RATING'].values[0],
            'pace': team_stats['PACE'].values[0],
            'win_pct': team_stats['W_PCT'].values[0]
        }
    except:
        return {}

def get_game_boxscore_stats(game_id):
    """Meccs boxscore statisztikák"""
    try:
        bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        data = bs.get_dict()
        
        home = data['boxScoreTraditional']['homeTeam']['statistics']
        away = data['boxScoreTraditional']['awayTeam']['statistics']
        
        return {
            'home_fg_pct': home.get('fieldGoalsPercentage', 0),
            'away_fg_pct': away.get('fieldGoalsPercentage', 0),
            'home_three_pct': home.get('threePointersPercentage', 0),
            'away_three_pct': away.get('threePointersPercentage', 0),
            'home_ft_pct': home.get('freeThrowsPercentage', 0),
            'away_ft_pct': away.get('freeThrowsPercentage', 0),
            'home_rebounds': home.get('reboundsTotal', 0),
            'away_rebounds': away.get('reboundsTotal', 0),
            'home_turnovers': home.get('turnovers', 0),
            'away_turnovers': away.get('turnovers', 0),
            'home_team_id': data['boxScoreTraditional']['homeTeam']['teamId'],
            'away_team_id': data['boxScoreTraditional']['awayTeam']['teamId']
        }
    except:
        return {}

def calculate_recent_momentum(actions, snapshot_period, snapshot_time):
    """Momentum az elmúlt 5 percben"""
    
    target_time_total = (snapshot_period - 1) * 720 + snapshot_time
    lookback_time = target_time_total - 300  # 5 perc
    
    # Snapshot időpontban a score
    current_home = 0
    current_away = 0
    
    # 5 perccel ezelőtti score
    lookback_home = 0
    lookback_away = 0
    
    for action in actions:
        if not action.get('scoreHome') or not action.get('scoreAway'):
            continue
            
        action_period = action.get('period', 1)
        clock = parse_clock(action.get('clock', 'PT00M00S'))
        action_time_total = (action_period - 1) * 720 + clock
        
        # Snapshot időponthoz legközelebbi score
        if abs(action_time_total - target_time_total) < 30:  # 30 mp tolerancia
            current_home = int(action['scoreHome'])
            current_away = int(action['scoreAway'])
        
        # 5 perccel ezelőtti score (legközelebbi)
        if abs(action_time_total - lookback_time) < 30:
            lookback_home = int(action['scoreHome'])
            lookback_away = int(action['scoreAway'])
    
    # Utolsó 5 percben szerzett pontok
    home_points_last_5min = current_home - lookback_home
    away_points_last_5min = current_away - lookback_away
    
    return {
        'home_points_last_5min': max(0, home_points_last_5min),  # Negatív ne legyen
        'away_points_last_5min': max(0, away_points_last_5min),
        'momentum_diff': home_points_last_5min - away_points_last_5min
    }

def enrich_snapshots(snapshots_csv, output_csv='ml_ready.csv', season='2024-25'):
    """Snapshots kiegészítése feature-ökkel"""
    
    # Már feldolgozott sorok
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        processed = set(zip(existing['game_id'], existing['period'], existing['time_remaining_period']))
        print(f"Már feldolgozott snapshots: {len(processed)}")
    else:
        existing = pd.DataFrame()
        processed = set()
    
    snapshots = pd.read_csv(snapshots_csv)
    enriched_rows = []
    
    current_game_id = None
    game_cache = {}
    
    for idx, row in snapshots.iterrows():
        snapshot_key = (str(row['game_id']), row['period'], row['time_remaining_period'])
        
        if snapshot_key in processed:
            continue
        
        print(f"[{idx+1}/{len(snapshots)}] Enriching: {row['game_id']}, Q{row['period']}, {row['time_remaining_period']}s")
        
        try:
            game_id = f"00{row['game_id']}"
            
            # Boxscore stats (cache-elve game_id szerint)
            if current_game_id != game_id:
                game_cache = get_game_boxscore_stats(game_id)
                
                # Play-by-play betöltése
                pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, start_period='1', end_period='4')
                actions = pbp.get_dict()['game']['actions']
                game_cache['actions'] = actions
                
                current_game_id = game_id
                time.sleep(0.6)
            
            # Momentum számítás
            momentum = calculate_recent_momentum(
                game_cache.get('actions', []),
                row['period'],
                row['time_remaining_period']
            )
            
            # Feature összegyűjtése
            enriched = {
                'game_id': game_id,
                'period': row['period'],
                'time_remaining_period': row['time_remaining_period'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'score_diff': row['score_diff'],
                **momentum,
            }
            
            enriched_rows.append(enriched)
            
            # Mentés minden 10. sor után
            if len(enriched_rows) >= 10:
                new_df = pd.DataFrame(enriched_rows)
                if len(existing) > 0:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df
                
                combined.to_csv(output_csv, index=False)
                existing = combined
                enriched_rows = []
                
                for er in new_df.to_dict('records'):
                    processed.add((str(er['game_id']), er['period'], er['time_remaining_period']))
            
        except Exception as e:
            print(f"Hiba: {e}")
            time.sleep(2)
            continue
    
    # Utolsó sorok mentése
    if len(enriched_rows) > 0:
        new_df = pd.DataFrame(enriched_rows)
        if len(existing) > 0:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(output_csv, index=False)
    
    print(f"Kész! ML-ready adatok: {output_csv}")

if __name__ == "__main__":
    enrich_snapshots('data/snapshots_2024_25.csv', 'data/ml_ready_2024_25.csv', '2024-25')