"""
Configuration settings for Player Tracker
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # Directories
    DATA_DIR = BASE_DIR / "data"
    IMPORTS_DIR = DATA_DIR / "imports"
    # API
    API_PORT = 8003
    API_HOST = "localhost"
    API_VERSION = "v1"
    
    # Database
    DATABASE_URL = f"sqlite:///{BASE_DIR / 'data'}/player_tracker.db"
    
    # ClubElo API
    CLUBELO_API_URL = "http://api.clubelo.com"
    CLUBELO_MANUAL_UPDATE = True  # Manual trigger via script/endpoint
    
    # Evaluation
    EVALUATION_VERSIONS = ["standard", "attack_focused", "defense_focused"]
    FORM_TREND_WINDOW = 10  # last N matches
    FORM_TREND_ALPHA = 0.3  # EMA smoothing factor
    
    # Percentile calculation
    MIN_MINUTES_FOR_EVALUATION = 450  # ~5 matches
    EXPECTED_TRACKED_PLAYERS = 50  # Performance optimization target
    
    # Notifications
    ENABLE_NOTIFICATIONS = False  # Disabled, using manual refresh instead
    
    # Import
    PLAYER_MATCH_THRESHOLD = 80  # fuzzy matching score
    AUTO_CREATE_PLAYERS = False  # create new players on import
    
    # Position mappings
    POSITION_GROUPS = {
        "GK": ["GK", "Goalkeeper"],
        "DEF": ["DF", "CB", "LB", "RB", "LWB", "RWB", "Defender"],
        "MID": ["MF", "CM", "DM", "AM", "LM", "RM", "Midfielder"],
        "ATT": ["FW", "ST", "CF", "LW", "RW", "Attacker", "Forward"]
    }
    
    # Stat weights by position for composite score
    STAT_WEIGHTS = {
        "ATT": {
            "attacking": 1.0,
            "defensive": 0.1,
            "possession": 0.6,
            "efficiency": 0.8,
            "consistency": 0.5
        },
        "MID": {
            "attacking": 0.6,
            "defensive": 0.5,
            "possession": 1.0,
            "efficiency": 0.7,
            "consistency": 0.6
        },
        "DEF": {
            "attacking": 0.2,
            "defensive": 1.0,
            "possession": 0.5,
            "efficiency": 0.6,
            "consistency": 0.7
        },
        "GK": {
            "attacking": 0.0,
            "defensive": 1.0,
            "possession": 0.3,
            "efficiency": 0.9,
            "consistency": 0.8
        }
    }

config = Config()
