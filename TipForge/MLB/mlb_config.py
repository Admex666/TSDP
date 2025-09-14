# mlb_config.py
MLB_TEAM_IDS = {
    # American League
    108: {'abbreviation': 'LAA', 'name': 'Los Angeles Angels', 'league': 'AL', 'division': 'West'},
    109: {'abbreviation': 'ARI', 'name': 'Arizona Diamondbacks', 'league': 'NL', 'division': 'West'},
    110: {'abbreviation': 'BAL', 'name': 'Baltimore Orioles', 'league': 'AL', 'division': 'East'},
    111: {'abbreviation': 'BOS', 'name': 'Boston Red Sox', 'league': 'AL', 'division': 'East'},
    112: {'abbreviation': 'CHC', 'name': 'Chicago Cubs', 'league': 'NL', 'division': 'Central'},
    113: {'abbreviation': 'CIN', 'name': 'Cincinnati Reds', 'league': 'NL', 'division': 'Central'},
    114: {'abbreviation': 'CLE', 'name': 'Cleveland Guardians', 'league': 'AL', 'division': 'Central'},
    115: {'abbreviation': 'COL', 'name': 'Colorado Rockies', 'league': 'NL', 'division': 'West'},
    116: {'abbreviation': 'DET', 'name': 'Detroit Tigers', 'league': 'AL', 'division': 'Central'},
    117: {'abbreviation': 'HOU', 'name': 'Houston Astros', 'league': 'AL', 'division': 'West'},
    118: {'abbreviation': 'KCR', 'name': 'Kansas City Royals', 'league': 'AL', 'division': 'Central'},
    119: {'abbreviation': 'LAD', 'name': 'Los Angeles Dodgers', 'league': 'NL', 'division': 'West'},
    120: {'abbreviation': 'WSN', 'name': 'Washington Nationals', 'league': 'NL', 'division': 'East'},
    121: {'abbreviation': 'NYM', 'name': 'New York Mets', 'league': 'NL', 'division': 'East'},
    133: {'abbreviation': 'OAK', 'name': 'Oakland Athletics', 'league': 'AL', 'division': 'West'},
    134: {'abbreviation': 'PIT', 'name': 'Pittsburgh Pirates', 'league': 'NL', 'division': 'Central'},
    135: {'abbreviation': 'SDP', 'name': 'San Diego Padres', 'league': 'NL', 'division': 'West'},
    136: {'abbreviation': 'SEA', 'name': 'Seattle Mariners', 'league': 'AL', 'division': 'West'},
    137: {'abbreviation': 'SFG', 'name': 'San Francisco Giants', 'league': 'NL', 'division': 'West'},
    138: {'abbreviation': 'STL', 'name': 'St. Louis Cardinals', 'league': 'NL', 'division': 'Central'},
    139: {'abbreviation': 'TBR', 'name': 'Tampa Bay Rays', 'league': 'AL', 'division': 'East'},
    140: {'abbreviation': 'TEX', 'name': 'Texas Rangers', 'league': 'AL', 'division': 'West'},
    141: {'abbreviation': 'TOR', 'name': 'Toronto Blue Jays', 'league': 'AL', 'division': 'East'},
    142: {'abbreviation': 'MIN', 'name': 'Minnesota Twins', 'league': 'AL', 'division': 'Central'},
    143: {'abbreviation': 'PHI', 'name': 'Philadelphia Phillies', 'league': 'NL', 'division': 'East'},
    144: {'abbreviation': 'ATL', 'name': 'Atlanta Braves', 'league': 'NL', 'division': 'East'},
    145: {'abbreviation': 'CHW', 'name': 'Chicago White Sox', 'league': 'AL', 'division': 'Central'},
    146: {'abbreviation': 'MIA', 'name': 'Miami Marlins', 'league': 'NL', 'division': 'East'},
    147: {'abbreviation': 'NYY', 'name': 'New York Yankees', 'league': 'AL', 'division': 'East'},
    158: {'abbreviation': 'MIL', 'name': 'Milwaukee Brewers', 'league': 'NL', 'division': 'Central'}
}

MLB_TEAM_IDS_BY_ABBR = {v['abbreviation']: {'id': k, **v} for k, v in MLB_TEAM_IDS.items()}
MLB_TEAM_IDS_BY_NAME = {v['name']: {'id': k, **v} for k, v in MLB_TEAM_IDS.items()}

# Team name mapping from API to Tippmix
API_TO_TIPPMIX = {
    "Tampa Bay Rays": "Tampa Bay",
    "Baltimore Orioles": "Baltimore",
    "Pittsburgh Pirates": "Pittsburgh",
    "Texas Rangers": "Texas",
    "Detroit Tigers": "Detroit",
    "New York Yankees": "NY Yankees",
    "Kansas City Royals": "Kansas City",
    "Chicago White Sox": "Chicago WS",
    "Arizona Diamondbacks": "Arizona",
    "Houston Astros": "Houston",
    "St. Louis Cardinals": "St. Louis",
    "Colorado Rockies": "Colorado",
    "Los Angeles Dodgers": "LA Dodgers",
    "Los Angeles Angels": "LA Angels",
    "Cincinnati Reds": "Cincinnati",
    "Chicago Cubs": "Chicago Cubs",
    "Toronto Blue Jays": "Toronto",
    "Washington Nationals": "Washington",
    "New York Mets": "NY Mets",
    "Miami Marlins": "Miami",
    "Boston Red Sox": "Boston",
    "Philadelphia Phillies": "Philadelphia",
    "Cleveland Guardians": "Cleveland",
    "Minnesota Twins": "Minnesota",
    "Atlanta Braves": "Atlanta",
    "Milwaukee Brewers": "Milwaukee",
    "San Diego Padres": "San Diego",
    "San Francisco Giants": "San Francisco",
    "Seattle Mariners": "Seattle",
    "Athletics": "Las Vegas",
}

THRESHOLD = 0.05
MODEL_PATH = 'models/'

# Google Sheets configuration
GOOGLE_SHEETS_ID = "your_google_sheets_id_here"
GOOGLE_CREDENTIALS_FILE = "credentials.json"