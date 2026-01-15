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
        # Cache for game start timestamps (game_id -> start_rfc460Timestamp)
        self.game_start_times = {}

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

    def _ensure_game_start_time(self, game_id):
        """
        Ensures we have the game start timestamp cached.
        Fetches the first frame if not already cached.
        Returns the start timestamp or None if unavailable.
        """
        if game_id in self.game_start_times:
            return self.game_start_times[game_id]
        
        # Fetch window without startingTime to get early frames
        try:
            window = self.get_live_stats_window(game_id, starting_time=None)
            if window and window.get("frames") and len(window["frames"]) > 0:
                # Use the first frame's timestamp as the game start
                first_frame = window["frames"][0]
                start_timestamp = first_frame.get("rfc460Timestamp")
                if start_timestamp:
                    self.game_start_times[game_id] = start_timestamp
                    logger.info(f"Cached game start time for {game_id}: {start_timestamp}")
                    return start_timestamp
        except Exception as e:
            logger.error(f"Error fetching game start time: {e}")
        
        return None

    def get_latest_match_state(self, game_id):
        """
        Fetches the latest state and formats it for the ValueBettingEngine.
        Uses AndyDanger's method: calculates game time from rfc460Timestamp difference.
        """
        starting_time = self._get_js_starting_time()
        
        # Ensure we have the game start time
        start_timestamp = self._ensure_game_start_time(game_id)
        
        # Fetch both window and details with the calculated starting time
        window = self.get_live_stats_window(game_id, starting_time=starting_time)
        details = self.get_live_stats_details(game_id, starting_time=starting_time)
        
        if not window or not window.get("frames"):
            return None
            
        latest_window = window["frames"][-1]
        
        latest_details = None
        if details and details.get("frames"):
            latest_details = details["frames"][-1]
            
        blue = latest_window.get("blueTeam", {})
        red = latest_window.get("redTeam", {})
        
        # Calculate game time using AndyDanger's method
        game_time_str = "00:00"
        
        if start_timestamp:
            try:
                current_timestamp = latest_window.get("rfc460Timestamp")
                if current_timestamp:
                    # Parse timestamps
                    start_dt = datetime.fromisoformat(start_timestamp.replace("Z", "+00:00"))
                    current_dt = datetime.fromisoformat(current_timestamp.replace("Z", "+00:00"))
                    
                    # Calculate difference in seconds
                    duration_seconds = int((current_dt - start_dt).total_seconds())
                    
                    # Format as MM:SS or HH:MM:SS
                    hours = duration_seconds // 3600
                    minutes = (duration_seconds % 3600) // 60
                    seconds = duration_seconds % 60
                    
                    if hours > 0:
                        game_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                    else:
                        game_time_str = f"{minutes:02d}:{seconds:02d}"
            except Exception as e:
                logger.error(f"Error calculating game time: {e}")
                game_time_str = "LIVE"
        else:
            # Fallback if we don't have start time yet
            game_time_str = "LIVE"
        
        formatted = {
            'timestamp': datetime.now().isoformat(),
            'game_time': game_time_str,
            'blue_team': {
                'kills': blue.get('totalKills', 0),
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
            'players': []
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
