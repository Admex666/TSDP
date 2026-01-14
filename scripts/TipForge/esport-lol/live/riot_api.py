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

    def get_latest_match_state(self, game_id):
        """
        Fetches the latest state and formats it for the ValueBettingEngine.
        """
        window = self.get_live_stats_window(game_id)
        details = self.get_live_stats_details(game_id)
        
        if not window or not window.get("frames"):
            return None
            
        latest_window = window["frames"][-1]
        latest_details = details["frames"][-1] if details and details.get("frames") else None
        
        # Calculate game time
        start_ts = window["frames"][0]["rfc460Timestamp"]
        now_ts = latest_window["rfc460Timestamp"]
        
        # Format like the legacy scraper output
        blue = latest_window.get("blueTeam", {})
        red = latest_window.get("redTeam", {})
        
        # Estimate game_time string "MM:SS"
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
            dt_now = datetime.strptime(now_ts, fmt)
            dt_start = datetime.strptime(start_ts, fmt)
            diff = dt_now - dt_start
            minutes = int(diff.total_seconds() // 60)
            seconds = int(diff.total_seconds() % 60)
            game_time_str = f"{minutes:02d}:{seconds:02d}"
        except:
            game_time_str = "15:00"

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
