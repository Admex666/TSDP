# mlb_data_loader.py
import joblib
import glob
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from mlb_config import MODEL_PATH, MLB_TEAM_IDS_BY_ABBR

def load_mlb_models():
    """MLB modellek betöltése"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
        model_files = glob.glob(os.path.join(model_path, 'mlb_simple_model_*.joblib'))
        
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            
            # Megfelelő scaler és features fájlok keresése
            timestamp = '_'.join(latest_model.split('_')[-2:]).replace('.joblib', '')
            scaler_path = os.path.join(model_path, f'mlb_simple_scaler_{timestamp}.joblib')
            features_path = os.path.join(model_path, f'mlb_simple_features_{timestamp}.joblib')
            
            model = joblib.load(latest_model)
            scaler = joblib.load(scaler_path)
            features = joblib.load(features_path)
            
            print(f"Model loaded: {latest_model}")
            return model, scaler, features, True
        else:
            print(f"No model files found in: {model_path}")
            return None, None, None, False
    except Exception as e:
        print(f"Model load error: {e}")
        return None, None, None, False

def get_upcoming_mlb_games(span=3):
    """
    Közelgő MLB mérkőzések lekérése
    """
    url = "https://statsapi.mlb.com/api/v1/schedule/games/"
    params = {
        'sportId': 1,  # MLB sport ID
        'startDate': datetime.now().strftime('%Y-%m-%d'),
        'endDate': (datetime.now() + timedelta(days=span)).strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        upcoming_games = []
        
        for date in data['dates']:
            for game in date['games']:
                game_info = {
                    'game_id': game['gamePk'],
                    'date': date['date'],
                    'game_datetime': game['gameDate'],
                    'away_team_id': game['teams']['away']['team']['id'],
                    'away_team_name': game['teams']['away']['team']['name'],
                    'home_team_id': game['teams']['home']['team']['id'],
                    'home_team_name': game['teams']['home']['team']['name'],
                    'venue': game['venue']['name']
                }
                upcoming_games.append(game_info)
        
        return pd.DataFrame(upcoming_games)
        
    except Exception as e:
        print(f"Error fetching MLB games: {e}")
        return pd.DataFrame()

def get_team_recent_stats(team_id, games=5):
    """
    Csapat utóbbi mérkőzéseinek statisztikái
    """
    try:
        # Season stats URL
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
        params = {
            'season': datetime.now().year,
            'stats': 'season',
            'group': 'hitting'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if not data['stats'] or not data['stats'][0]['splits']:
            print(f"No stats found for team {team_id}")
            return {
                'avg': 0.250,
                'obp': 0.320,
                'slg': 0.400,
                'games': 0
            }
        
        # Season totals
        season_stats = data['stats'][0]['splits'][0]['stat']
        
        return {
            'avg': float(season_stats.get('avg', 0.250)),
            'obp': float(season_stats.get('obp', 0.320)),
            'slg': float(season_stats.get('slg', 0.400)),
            'games': int(season_stats.get('gamesPlayed', 0))
        }
        
    except Exception as e:
        print(f"Error getting stats for team {team_id}: {e}")
        return {
            'avg': 0.250,
            'obp': 0.320,
            'slg': 0.400,
            'games': 0
        }