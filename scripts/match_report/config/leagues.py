"""
Configuration for supported leagues and tournaments
"""

LEAGUES = {
    "Premier League": {
        "tournament_id": 17,
        "season_id": 76986,
        "name": "Premier League",
        "country": "England",
        "current_round": 23,
        "total_rounds": 38
    },
    "La Liga": {
        "tournament_id": 8,
        "season_id": 61643,
        "name": "LaLiga",
        "country": "Spain",
        "current_round": 20,
        "total_rounds": 38
    },
    "Bundesliga": {
        "tournament_id": 35,
        "season_id": 61728,
        "name": "Bundesliga",
        "country": "Germany",
        "current_round": 18,
        "total_rounds": 34
    },
    "Serie A": {
        "tournament_id": 23,
        "season_id": 61644,
        "name": "Serie A",
        "country": "Italy",
        "current_round": 21,
        "total_rounds": 38
    },
    "Ligue 1": {
        "tournament_id": 34,
        "season_id": 61648,
        "name": "Ligue 1",
        "country": "France",
        "current_round": 18,
        "total_rounds": 34
    },
    "Champions League": {
        "tournament_id": 7,
        "season_id": 76953,
        "name": "UEFA Champions League",
        "country": "Europe",
        "current_round": 7,
        "total_rounds": 8
    }
}

# Pitch coordinate constants
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68   # meters

# Shot location zones (based on 0-100 coordinate system)
PENALTY_BOX_X = 83  # Penalty box starts at ~83
SIX_YARD_BOX_X = 94  # 6-yard box starts at ~94
CENTRAL_ZONE_Y_MIN = 37
CENTRAL_ZONE_Y_MAX = 63

# Shot quality thresholds
HIGH_QUALITY_XG_PER_SHOT = 0.12
SPECULATIVE_SHOOTING_PCT = 30

# Pass network thresholds
MIN_PASSES_FOR_CONNECTION = 10
PROGRESSIVE_PASS_THRESHOLD = 10  # meters forward
BACKWARD_PASS_THRESHOLD = -5    # meters backward
