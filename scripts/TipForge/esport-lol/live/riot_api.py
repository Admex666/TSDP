import requests
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RiotEsportsAPI:
    """Connector for the official Riot Esports API (undocumented)"""
    
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL_GW = "https://esports-api.lolesports.com/persisted/gw"
    BASE_URL_FEED = "https://feed.lolesports.com/livestats/v1"
    
    def __init__(self):
        self.headers = {
            "x-api-key": self.API_KEY,
            "Accept": "application/json"
        }

    def get_schedule(self, hl="en-US"):
        """Fetch the current and upcoming schedule"""
        try:
            url = f"{self.BASE_URL_GW}/getSchedule"
            res = requests.get(url, params={"hl": hl}, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("schedule", {}).get("events", [])
            logger.error(f"Failed to fetch schedule: {res.status_code}")
        except Exception as e:
            logger.error(f"Error in get_schedule: {e}")
        return []

    def get_standings(self, tournament_id, hl="en-US"):
        """Fetch standings for a specific tournament"""
        try:
            url = f"{self.BASE_URL_GW}/getStandings"
            res = requests.get(url, params={"hl": hl, "tournamentId": tournament_id}, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("standings", [])
        except Exception as e:
            logger.error(f"Error in get_standings: {e}")
        return []

    def get_leagues(self, hl="en-US"):
        """Fetch all leagues"""
        try:
            url = f"{self.BASE_URL_GW}/getLeagues"
            res = requests.get(url, params={"hl": hl}, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("leagues", [])
        except Exception as e:
            logger.error(f"Error in get_leagues: {e}")
        return []

    def get_event_details(self, event_id, hl="en-US"):
        """Fetch details for a specific event (match)"""
        try:
            url = f"{self.BASE_URL_GW}/getEventDetails"
            res = requests.get(url, params={"hl": hl, "id": event_id}, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("event", {})
        except Exception as e:
            logger.error(f"Error in get_event_details: {e}")
        return None

    def get_live(self, hl="en-US"):
        """Fetch currently live matches"""
        try:
            url = f"{self.BASE_URL_GW}/getLive"
            res = requests.get(url, params={"hl": hl}, headers=self.headers)
            if res.status_code == 200:
                return res.json().get("data", {}).get("schedule", {}).get("events", [])
        except Exception as e:
            logger.error(f"Error in get_live: {e}")
        return []

    def get_live_stats_window(self, game_id, starting_time=None):
        """Fetch live stats window for a specific game"""
        try:
            url = f"{self.BASE_URL_FEED}/window/{game_id}"
            params = {}
            if starting_time:
                params["startingTime"] = starting_time
            res = requests.get(url, params=params)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.error(f"Error in get_live_stats_window: {e}")
        return None

    def get_live_stats_details(self, game_id, starting_time=None):
        """Fetch detailed live stats for a specific game"""
        try:
            url = f"{self.BASE_URL_FEED}/details/{game_id}"
            params = {}
            if starting_time:
                params["startingTime"] = starting_time
            res = requests.get(url, params=params)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.error(f"Error in get_live_stats_details: {e}")
        return None

    def _get_js_starting_time(self):
        """
        Calculates the startingTime parameter matching the logic used by live-lol-esports.
        Formula: (Now - (Now % 10s)) - 60s
        """
        now = datetime.now(os.sys.modules['datetime'].timezone.utc) if 'timezone' in dir(datetime) else datetime.utcnow()
        seconds = now.second
        rounded_seconds = seconds - (seconds % 10)
        dt_rounded = now.replace(second=rounded_seconds, microsecond=0)
        dt_final = dt_rounded - os.sys.modules['datetime'].timedelta(seconds=60)
        return dt_final.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def get_latest_match_state(self, game_id):
        """
        Fetches the latest state and formats it for the ValueBettingEngine.
        """
        starting_time = self._get_js_starting_time()
        
        # Fetch both window and details with the calculated starting time
        window = self.get_live_stats_window(game_id, starting_time=starting_time)
        details = self.get_live_stats_details(game_id, starting_time=starting_time)
        
        if not window or not window.get("frames"):
            # Fallback: try without starting time if it fails (e.g. game just started)
            # Or handle the case where the window is empty
            return None
            
        latest_window = window["frames"][-1]
        
        # For details, we might not get frames if the startingTime is too recent or different?
        # But we should try to match them.
        latest_details = None
        if details and details.get("frames"):
            latest_details = details["frames"][-1]
            
        # Calculate game time
        # The window frames contain `gameState` ('in_game') and timestamps.
        # We can try to use the timestamp from the frame.
        # But we need the START time of the game to calculate duration.
        # The first frame of the *window* is not the first frame of the game.
        # Use the logic from the user's JS: 
        # let k = function(e, t) { ... return formatted time ... }(FirstFrame.rfc460Timestamp, CurrentFrame.rfc460Timestamp)
        # But we don't have the "FirstFrame" of the game easily unless we fetch from the beginning once.
        # Ideally, we should cache the game start time.
        # For now, let's keep the estimation or try to find a better way.
        
        # Actually, `window['frames']` has `rfc460Timestamp`.
        # If we want "game_time", we need the difference between this timestamp and the game start.
        # We can fetch the *first* frame of the game (without startingTime) once to get the start time?
        # That sounds expensive to do every time.
        # Maybe just return the timestamp and let the scanner handle duration if it knows the start time.
        # OR: Accept that for now we might not have perfect game time, but we have perfect GOLD.
        
        now_ts = latest_window["rfc460Timestamp"]
        
        blue = latest_window.get("blueTeam", {})
        red = latest_window.get("redTeam", {})
        
        # Placeholder for game time if we can't calculate it perfectly yet
        game_time_str = "00:00" 
        # Attempt to parse timestamp to see if we can derive something?
        # Without game start time, we can't know the duration.
        
        formatted = {
            'timestamp': datetime.now().isoformat(),
            'game_time': game_time_str,
            'blue_team': {
                'kills': blue.get('totalKills', 0), # Note: API uses totalKills, not kills
                'towers': blue.get('towers', 0),
                'inhibitors': blue.get('inhibitors', 0),
                'barons': blue.get('barons', 0),
                'gold': blue.get('totalGold', 0),
                'dragons': blue.get('dragons', [])
            },
            'red_team': {
                'kills': red.get('totalKills', 0),
                'towers': red.get('towers', 0),
                'inhibitors': red.get('inhibitors', 0),
                'barons': red.get('barons', 0),
                'gold': red.get('totalGold', 0),
                'dragons': red.get('dragons', [])
            },
            'players': [] # Map from details if available
        }

        formatted = {
            'timestamp': datetime.now().isoformat(),
            'game_time': game_time_str,
            'blue_team': {
                'kills': blue.get('kills', 0),
                'towers': blue.get('towers', 0),
                'inhibitors': blue.get('inhibitors', 0),
                'barons': blue.get('barons', 0),
                'gold': blue.get('totalGold', 0),
                'dragons': blue.get('dragons', [])
            },
            'red_team': {
                'kills': red.get('kills', 0),
                'towers': red.get('towers', 0),
                'inhibitors': red.get('inhibitors', 0),
                'barons': red.get('barons', 0),
                'gold': red.get('totalGold', 0),
                'dragons': red.get('dragons', [])
            },
            'players': [] # Map from details if available
        }
        
        if latest_details and "participants" in latest_details:
            for p in latest_details["participants"]:
                formatted['players'].append({
                    'participantId': p.get('participantId'),
                    'cs': p.get('creepScore', 0),
                    'gold': p.get('totalGoldEarned', 0),
                    'kills': p.get('kills', 0),
                    'deaths': p.get('deaths', 0),
                    'assists': p.get('assists', 0)
                })
        
        return formatted
