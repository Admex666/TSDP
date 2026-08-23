import pandas as pd
import json
import time
import random
from typing import Dict, Any, Optional, List, Union
import tls_client
from functools import lru_cache

# Global session pool for reuse
_session_pool = []
_last_request_time = 0
_request_count = 0

# User-Agent rotation pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Client identifier rotation
CLIENT_IDENTIFIERS = [
    "chrome_120",
    "chrome_119", 
    "chrome_118",
    "safari_ios_16_0",
    "firefox_120",
]

def _get_session():
    """Get or create a TLS session from the pool"""
    global _session_pool
    
    if not _session_pool or random.random() < 0.3:  # 30% chance to create new session
        client_id = random.choice(CLIENT_IDENTIFIERS)
        sess = tls_client.Session(client_identifier=client_id)
        _session_pool.append(sess)
        
        # Keep pool size manageable
        if len(_session_pool) > 5:
            _session_pool.pop(0)
    
    return random.choice(_session_pool)

def _rate_limit():
    """Implement rate limiting to avoid triggering anti-bot measures"""
    global _last_request_time, _request_count
    
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    # Adaptive delay based on request count
    if _request_count > 10:
        min_delay = 2.0  # Slow down after many requests
    elif _request_count > 5:
        min_delay = 1.5
    else:
        min_delay = 1.0
    
    # Add random jitter to appear more human
    delay = min_delay + random.uniform(0.5, 1.5)
    
    if time_since_last < delay:
        sleep_time = delay - time_since_last
        time.sleep(sleep_time)
    
    _last_request_time = time.time()
    _request_count += 1
    
    # Reset counter periodically
    if _request_count > 20:
        _request_count = 0
        time.sleep(random.uniform(3, 5))  # Longer break

def _get_headers(url: str, referer: str = "https://www.sofascore.com/") -> Dict[str, str]:
    """Generate realistic headers with rotation"""
    user_agent = random.choice(USER_AGENTS)
    
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": referer,
        "Origin": "https://www.sofascore.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    
    # Add realistic sec-ch-ua headers for Chrome
    if "Chrome" in user_agent:
        headers["sec-ch-ua"] = '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"'
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'
    
    return headers

def scrape_sofascore(url: str, max_retries: int = 10, referer: str = "https://www.sofascore.com/") -> Dict[str, Any]:
    """
    Scrape SofaScore API with advanced anti-403 protection
    
    Args:
        url: API endpoint URL
        max_retries: Maximum number of retry attempts
        referer: Referer URL to use in headers
        
    Returns:
        JSON response as dictionary, empty dict on failure
    """
    _rate_limit()
    
    for attempt in range(max_retries):
        try:
            # Get session and headers
            sess = _get_session()
            headers = _get_headers(url, referer=referer)
            
            # Add small random delay before request
            time.sleep(random.uniform(0.2, 0.5))
            
            # Make request
            resp = sess.get(url, headers=headers)
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    return data
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    if attempt < max_retries - 1:
                        continue
                    return {}
            
            elif resp.status_code == 403:
                # print(f"403 Forbidden (attempt {attempt + 1}/{max_retries})")
                
                # Exponential backoff with more jitter and higher base
                wait_time = (3 ** (attempt // 2)) + random.uniform(2, 5)
                # print(f"Waiting {wait_time:.2f}s before retry...")
                time.sleep(wait_time)
                
                # Clear session pool periodically
                global _session_pool
                if len(_session_pool) > 0:
                    _session_pool = []
                
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"Failed after {max_retries} attempts: {url}")
                    return {}
            
            elif resp.status_code == 429:  # Too Many Requests
                print(f"Rate limited (429), waiting longer...")
                wait_time = 10 + random.uniform(5, 10)
                time.sleep(wait_time)
                
                if attempt < max_retries - 1:
                    continue
                else:
                    return {}
            
            elif resp.status_code == 404:
                print(f"Resource not found (404): {url}")
                return {}
            
            else:
                print(f"Error {resp.status_code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {}
                    
        except Exception as e:
            print(f"Request exception (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {}
    
    return {}


# 1. Lineups DataFrame
def create_lineups_df(event_id):
    home_players = []
    away_players = []
    
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/lineups"
    lineups_data = scrape_sofascore(url)
    
    # Process home players
    for player in lineups_data['home']['players']:
        player_info = player['player'].copy()
        if 'statistics' in player:
            player_info.update(player['statistics'])
        player_info['team'] = 'home'
        player_info['substitute'] = player['substitute']
        player_info['captain'] = player.get('captain', False)
        home_players.append(player_info)
    
    # Process away players
    for player in lineups_data['away']['players']:
        player_info = player['player'].copy()
        if 'statistics' in player:
            player_info.update(player['statistics'])
        player_info['team'] = 'away'
        player_info['substitute'] = player['substitute']
        player_info['captain'] = player.get('captain', False)
        away_players.append(player_info)
    
    # Combine both teams
    all_players = home_players + away_players
    lineups_df = pd.DataFrame(all_players)
    
    return lineups_df

# 2. Average Positions DataFrame
def create_average_positions_df(event_id):
    home_positions = []
    away_positions = []

    url = f"https://www.sofascore.com/api/v1/event/{event_id}/average-positions"
    average_positions_data = scrape_sofascore(url)
    
    # Process home players
    for player in average_positions_data['home']:
        player_info = player['player'].copy()
        player_info.update({
            'averageX': player['averageX'],
            'averageY': player['averageY'],
            'pointsCount': player['pointsCount'],
            'team': 'home'
        })
        home_positions.append(player_info)
    
    # Process away players
    for player in average_positions_data['away']:
        player_info = player['player'].copy()
        player_info.update({
            'averageX': player['averageX'],
            'averageY': player['averageY'],
            'pointsCount': player['pointsCount'],
            'team': 'away'
        })
        away_positions.append(player_info)
    
    # Combine both teams
    all_positions = home_positions + away_positions
    positions_df = pd.DataFrame(all_positions)
    
    return positions_df

# 3. Statistics DataFrame
def create_statistics_df(event_id):
    all_stats = []

    url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
    statistics_data = scrape_sofascore(url)
    
    for period_data in statistics_data['statistics']:
        period = period_data['period']
        
        for group in period_data['groups']:
            group_name = group['groupName']
            
            for item in group['statisticsItems']:
                stat_info = {
                    'period': period,
                    'group': group_name,
                    'statistic': item['name'],
                    'home_value': item.get('homeValue'),
                    'away_value': item.get('awayValue'),
                    'home_total': item.get('homeTotal'),
                    'away_total': item.get('awayTotal'),
                    'home_display': item.get('home'),
                    'away_display': item.get('away'),
                    'compare_code': item.get('compareCode'),
                    'key': item.get('key')
                }
                all_stats.append(stat_info)
    
    return pd.DataFrame(all_stats)

# 4. Shotmap DataFrame
def create_shotmap_df(event_id):
    shots = []

    url = f"https://www.sofascore.com/api/v1/event/{event_id}/shotmap"
    shotmap_data = scrape_sofascore(url)
    
    for shot in shotmap_data['shotmap']:
        shot_info = shot['player'].copy()
        shot_info.update({
            'isHome': shot['isHome'],
            'shotType': shot['shotType'],
            'situation': shot['situation'],
            'bodyPart': shot['bodyPart'],
            'time': shot['time'],
            'timeSeconds': shot['timeSeconds'],
            'periodTimeSeconds': shot.get('periodTimeSeconds'),
            'goalType': shot.get('goalType'),
            'playerX': shot['playerCoordinates']['x'],
            'playerY': shot['playerCoordinates']['y'],
            'goalMouthX': shot['goalMouthCoordinates']['x'] if 'goalMouthCoordinates' in shot else None,
            'goalMouthY': shot['goalMouthCoordinates']['y'] if 'goalMouthCoordinates' in shot else None,
            'goalMouthZ': shot['goalMouthCoordinates']['z'] if 'goalMouthCoordinates' in shot else None
        })
        shots.append(shot_info)
    
    return pd.DataFrame(shots)

# 5. Graph DataFrame
def create_graph_df(event_id):
    graph_points = []

    url = f"https://www.sofascore.com/api/v1/event/{event_id}/graph"
    graph_data = scrape_sofascore(url)
    
    for point in graph_data['graphPoints']:
        graph_points.append({
            'minute': point['minute'],
            'value': point['value']
        })
    
    return pd.DataFrame(graph_points)

# 6. Player stats DataFrame
def create_player_stats_df(player_stats_data):
    """
    Convert player statistics data to a pandas DataFrame
    
    Parameters:
    player_stats_data (list): List of player statistics dictionaries
    
    Returns:
    pd.DataFrame: DataFrame containing player statistics
    """
    all_player_stats = []
    
    for player_data in player_stats_data:
        # Alap játékos információk
        player_info = player_data['player'].copy()
        
        # Csapat információk hozzáadása
        player_info['team_name'] = player_data['team']['name']
        player_info['team_id'] = player_data['team']['id']
        
        # Pozíció hozzáadása
        player_info['position'] = player_data.get('position', '')
        
        # Statisztikák hozzáadása, ha vannak
        if player_data['statistics']:
            player_info.update(player_data['statistics'])
        
        all_player_stats.append(player_info)
    
    # DataFrame létrehozása
    player_stats_df = pd.DataFrame(all_player_stats)
    
    return player_stats_df

# 7. Game odds
def create_odds_df(event_id):
    odds_list = []

    url = f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all"
    odds_data = scrape_sofascore(url)

    if odds_data.get("markets"):
        for market in odds_data['markets']:
            if (market['marketGroup'] == '1X2') and (market['marketName'] == 'Full time'):
                for choice in market['choices']:
                    dividend, divisor = choice["fractionalValue"].split('/') 
                    odds = int(dividend) / int(divisor) + 1
                    odds_list.append({
                        "name": choice["name"],
                        "odds": odds,
                        "prob": 1/odds
                    })

        df_odds = pd.DataFrame(odds_list)
        df_odds['prob_corr'] = df_odds['prob'] / df_odds['prob'].sum()
        
        return df_odds
    else:
        return pd.DataFrame()
    

def fetch_passmap(event_id, player_id):
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/player/{player_id}/rating-breakdown"

    resp = scrape_sofascore(url)
    rows = []

    for event_type, events in resp.items():
        for e in events:
            row = {
                "event_group": event_type,
                "eventActionType": e.get("eventActionType"),
                "isHome": e.get("isHome"),
                "outcome": e.get("outcome"),
                "player_x": e.get("playerCoordinates", {}).get("x"),
                "player_y": e.get("playerCoordinates", {}).get("y"),
                "end_x": e.get("passEndCoordinates", {}).get("x") if "passEndCoordinates" in e else None,
                "end_y": e.get("passEndCoordinates", {}).get("y") if "passEndCoordinates" in e else None,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def fetch_match_incidents(event_id: int, referer: str = "https://www.sofascore.com/") -> pd.DataFrame:
    """
    Fetch match incidents (goals, cards, substitutions, etc.) for a given event ID.
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/incidents"
    data = scrape_sofascore(url, referer=referer)
    
    if not data or 'incidents' not in data:
        return pd.DataFrame()
        
    incidents = []
    for incident in data['incidents']:
        incident_type = incident.get('incidentType')
        player_name = incident.get('player', {}).get('name')
        player_in_name = incident.get('playerIn', {}).get('name')
        player_out_name = incident.get('playerOut', {}).get('name')
        
        incidents.append({
            'time': incident.get('time'),
            'added_time': incident.get('addedTime'),
            'type': incident_type,
            'text': incident.get('text'),
            'is_home': incident.get('isHome'),
            'player_name': player_name,
            'player_in': player_in_name,
            'player_out': player_out_name,
            'incident_class': incident.get('incidentClass'),
            'description': incident.get('description'),
            'home_score': incident.get('homeScore'),
            'away_score': incident.get('awayScore')
        })
        
    return pd.DataFrame(incidents)


def fetch_match_details(event_id: int, referer: str = "https://www.sofascore.com/") -> Dict[str, Any]:
    """
    Fetch general match details (teams, status, scores, etc.) for a given event ID.
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}"
    data = scrape_sofascore(url, referer=referer)
    
    if not data or 'event' not in data:
        return {}
        
    event = data['event']
    
    details = {
        'match_id': event.get('id'),
        'home_team': event.get('homeTeam', {}).get('name'),
        'away_team': event.get('awayTeam', {}).get('name'),
        'home_score': event.get('homeScore', {}).get('display'),
        'away_score': event.get('awayScore', {}).get('display'),
        'status': event.get('status', {}).get('description'),
        'extra_time': event.get('time', {}).get('extra') is not None or 'After extra time' in event.get('status', {}).get('description', ''),
        'penalties': 'penalties' in event.get('status', {}).get('type', '') or 'After penalties' in event.get('status', {}).get('description', ''),
        'winner_code': event.get('winnerCode'),
        'start_timestamp': event.get('startTimestamp'),
    }
    
    return details


# 8. Match Comments / Play-by-Play DataFrame
def create_comments_df(event_id: int, referer: str = "https://www.sofascore.com/") -> pd.DataFrame:
    """
    Fetch play-by-play comments and micro-events for a given event ID and return as DataFrame.
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/comments"
    data = scrape_sofascore(url, referer=referer)
    
    if not data or 'comments' not in data:
        return pd.DataFrame()
        
    comments_list = []
    for c in data['comments']:
        player = c.get('player') or {}
        
        row = {
            'comment_id': c.get('id'),
            'time': c.get('time'),
            'period_name': c.get('periodName'),
            'type': c.get('type'),
            'text': c.get('text'),
            'is_home': c.get('isHome'),
            'player_id': player.get('id'),
            'player_name': player.get('name'),
            'player_slug': player.get('slug'),
            'player_position': player.get('position'),
            'player_jersey': player.get('jerseyNumber'),
        }
        comments_list.append(row)
        
    df = pd.DataFrame(comments_list)
    # Sort chronologically if time is present
    if not df.empty and 'time' in df.columns:
        df = df.sort_values(by=['time', 'comment_id'], ascending=[True, True]).reset_index(drop=True)
        
    return df


def fetch_match_comments(event_id: int, referer: str = "https://www.sofascore.com/") -> pd.DataFrame:
    """
    Alias for create_comments_df to maintain naming consistency with other fetch_* functions.
    """
    return create_comments_df(event_id, referer=referer)


# 9. All Match Passes & Ball Movements DataFrame
def create_all_passes_df(event_id: int, referer: str = "https://www.sofascore.com/") -> pd.DataFrame:
    """
    Fetch and aggregate all passes, ball-carries, and defensive actions for all players in a match.
    
    Returns:
        pd.DataFrame: Table containing all ball actions with player info, start (x, y) and end (x, y) coordinates,
                      outcome, and keypass indicators.
    """
    lineups_url = f"https://www.sofascore.com/api/v1/event/{event_id}/lineups"
    lineups_data = scrape_sofascore(lineups_url, referer=referer)
    
    if not lineups_data or ('home' not in lineups_data and 'away' not in lineups_data):
        return pd.DataFrame()
        
    all_players = []
    for side in ['home', 'away']:
        if side in lineups_data:
            for p in lineups_data[side].get('players', []):
                p_info = p.get('player', {})
                all_players.append({
                    'id': p_info.get('id'),
                    'name': p_info.get('name'),
                    'team': side,
                    'is_substitute': p.get('substitute', False)
                })
                
    all_events = []
    for player in all_players:
        pid = player['id']
        if not pid:
            continue
            
        rb_url = f"https://www.sofascore.com/api/v1/event/{event_id}/player/{pid}/rating-breakdown"
        rb_data = scrape_sofascore(rb_url, referer=referer)
        
        if not rb_data or not isinstance(rb_data, dict):
            continue
            
        for category, events in rb_data.items():
            if isinstance(events, list):
                for e in events:
                    row = {
                        'event_category': category,
                        'event_action_type': e.get('eventActionType'),
                        'player_id': pid,
                        'player_name': player['name'],
                        'team': player['team'],
                        'is_home': e.get('isHome'),
                        'outcome': e.get('outcome'),
                        'keypass': e.get('keypass', False),
                        'start_x': e.get('playerCoordinates', {}).get('x') if 'playerCoordinates' in e else None,
                        'start_y': e.get('playerCoordinates', {}).get('y') if 'playerCoordinates' in e else None,
                        'end_x': e.get('passEndCoordinates', {}).get('x') if 'passEndCoordinates' in e else None,
                        'end_y': e.get('passEndCoordinates', {}).get('y') if 'passEndCoordinates' in e else None,
                    }
                    all_events.append(row)
                    
    return pd.DataFrame(all_events)


def fetch_match_passes(event_id: int, referer: str = "https://www.sofascore.com/") -> pd.DataFrame:
    """
    Alias for create_all_passes_df.
    """
    return create_all_passes_df(event_id, referer=referer)


# 10. Live Match Stream Recorder & Parser
def record_live_match_stream(event_id: Union[int, str], duration_seconds: int = 5400, output_file: Optional[str] = None, match_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Record real-time live match stream (Sportradar timeline, ball coordinates, possession, micro-events).
    Uses headless Chrome CDP to capture the WebSocket data stream.
    
    Args:
        event_id: SofaScore match ID (or URL)
        duration_seconds: How long to record in seconds (default 5400s = 90 mins)
        output_file: Optional path to save JSON output
        match_url: Optional explicit SofaScore match URL
        
    Returns:
        List of captured raw/parsed event records
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        raise ImportError("Selenium is required for live stream recording. Install with: pip install selenium")
        
    import base64
    
    target_url = match_url
    clean_id = str(event_id)
    
    if isinstance(event_id, str) and ("sofascore.com" in event_id or "http" in event_id):
        target_url = event_id
        if "#id:" in event_id:
            clean_id = event_id.split("#id:")[-1]
    
    if not target_url:
        try:
            event_info = scrape_sofascore(f"https://www.sofascore.com/api/v1/event/{clean_id}")
            if event_info and "event" in event_info:
                slug = event_info["event"].get("slug")
                custom_id = event_info["event"].get("customId")
                if slug and custom_id:
                    target_url = f"https://www.sofascore.com/hu/football/match/{slug}/{custom_id}#id:{clean_id}"
        except Exception:
            pass
            
    if not target_url:
        target_url = f"https://www.sofascore.com/hu/football/match/match/#id:{clean_id}"
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(target_url)
        time.sleep(3)
        
        recorded_events = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            logs = driver.get_log("performance")
            for entry in logs:
                try:
                    log = json.loads(entry["message"])["message"]
                    method = log.get("method")
                    
                    if method == "Network.webSocketFrameReceived":
                        payload = log.get("params", {}).get("response", {}).get("payloadData", "")
                        timestamp = log.get("params", {}).get("timestamp", time.time())
                        
                        if payload:
                            try:
                                decoded_bytes = base64.b64decode(payload)
                                decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
                            except Exception:
                                decoded_text = payload
                                
                            entry_data = {
                                "timestamp": timestamp,
                                "time_str": time.strftime("%H:%M:%S"),
                                "raw_payload": decoded_text
                            }
                            recorded_events.append(entry_data)
                except Exception:
                    continue
                    
            time.sleep(0.5)
            
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(recorded_events, f, ensure_ascii=False, indent=2)
                
        return recorded_events
        
    finally:
        driver.quit()


def parse_live_stream_to_df(stream_data_or_file: Union[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Parse captured live stream data into a structured pandas DataFrame.
    
    Args:
        stream_data_or_file: File path string to JSON or list of captured raw stream dictionaries.
        
    Returns:
        pd.DataFrame with parsed real-time events, coordinates, situations, and timestamps.
    """
    import re
    if isinstance(stream_data_or_file, str):
        with open(stream_data_or_file, "r", encoding="utf-8") as f:
            events_data = json.load(f)
    else:
        events_data = stream_data_or_file
        
    parsed_rows = []
    
    for ev in events_data:
        raw = ev.get('raw_payload', '')
        time_str = ev.get('time_str', '')
        
        if 'match_timelinedelta' in raw or 'match_timeline' in raw:
            json_matches = re.findall(r'\{.*\}', raw)
            for jm in json_matches:
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(jm):
                    while idx < len(jm) and jm[idx] != '{':
                        idx += 1
                    if idx >= len(jm):
                        break
                    try:
                        obj, end_idx = decoder.raw_decode(jm[idx:])
                        idx += end_idx
                        
                        data_entries = obj.get('data', []) if isinstance(obj, dict) else []
                        if isinstance(data_entries, dict):
                            data_entries = [data_entries]
                            
                        for d in data_entries:
                            if not isinstance(d, dict):
                                continue
                            events_inner = d.get('data', {}).get('events', []) if isinstance(d.get('data'), dict) else []
                            for sub_ev in events_inner:
                                if not isinstance(sub_ev, dict):
                                    continue
                                coords = sub_ev.get('coordinates', [])
                                start_x, start_y, end_x, end_y = None, None, None, None
                                
                                if coords and len(coords) >= 1:
                                    end_x = coords[0].get('X')
                                    end_y = coords[0].get('Y')
                                    if len(coords) >= 2:
                                        start_x = coords[-1].get('X')
                                        start_y = coords[-1].get('Y')
                                elif sub_ev.get('X') is not None and sub_ev.get('Y') is not None:
                                    end_x = sub_ev.get('X')
                                    end_y = sub_ev.get('Y')

                                row = {
                                    'time_captured': time_str,
                                    'match_minute': sub_ev.get('time') if sub_ev.get('time', -1) >= 0 else None,
                                    'match_seconds': sub_ev.get('seconds') if sub_ev.get('seconds', -1) >= 0 else None,
                                    'type': sub_ev.get('type'),
                                    'name': sub_ev.get('name'),
                                    'situation': sub_ev.get('situation'),
                                    'team': sub_ev.get('team') or (coords[0].get('team') if coords and isinstance(coords[0], dict) else None),
                                    'possession': sub_ev.get('name') == 'Ball possession',
                                    'x': end_x,
                                    'y': end_y,
                                    'start_x': start_x,
                                    'start_y': start_y,
                                    'trajectory_points': coords if coords else None,
                                    'event_id': sub_ev.get('_id'),
                                    'home_score': sub_ev.get('homeScore'),
                                    'away_score': sub_ev.get('awayScore'),
                                }
                                parsed_rows.append(row)
                    except Exception:
                        idx += 1
                    
    df = pd.DataFrame(parsed_rows)
    if not df.empty and 'event_id' in df.columns:
        df = df.drop_duplicates(subset=['event_id']).reset_index(drop=True)
    return df


def fetch_live_match_events(event_id: Union[int, str], duration_seconds: int = 60, output_file: Optional[str] = None, match_url: Optional[str] = None) -> pd.DataFrame:
    """
    High-level function: Record live match stream for specified duration and return structured event DataFrame.
    
    Args:
        event_id: SofaScore match ID or URL
        duration_seconds: How long to record in seconds
        output_file: Optional path to save JSON raw stream
        match_url: Optional explicit match URL
        
    Returns:
        pd.DataFrame with parsed live events
    """
    events = record_live_match_stream(event_id=event_id, duration_seconds=duration_seconds, output_file=output_file, match_url=match_url)
    return parse_live_stream_to_df(events)

# 11. Match Live Events with Player Actions & Passes
def match_live_events_with_player_passes(
    live_events_df_or_path: Union[pd.DataFrame, str],
    match_id: Optional[Union[int, str]] = None,
    player_passes_df: Optional[pd.DataFrame] = None,
    max_coord_distance: float = 25.0
) -> pd.DataFrame:
    """
    Spatially and contextually match live WebSocket stream events with player passes, 
    ball-carries, and defensive actions from rating breakdown.
    
    Args:
        live_events_df_or_path: DataFrame or CSV path of recorded live stream events
        match_id: SofaScore match ID (required if player_passes_df is not provided)
        player_passes_df: Optional pre-fetched DataFrame of player passes (create_all_passes_df)
        max_coord_distance: Maximum coordinate distance threshold for matching
        
    Returns:
        pd.DataFrame containing enriched live events with player names, action types, outcome, and keypass flags.
    """
    import numpy as np
    
    if isinstance(live_events_df_or_path, str):
        df_live = pd.read_csv(live_events_df_or_path)
    else:
        df_live = live_events_df_or_path.copy()
        
    if df_live.empty:
        return pd.DataFrame()
        
    if player_passes_df is not None:
        df_passes = player_passes_df.copy()
    elif match_id:
        df_passes = create_all_passes_df(match_id)
    else:
        raise ValueError("Either player_passes_df or match_id must be provided.")
        
    if df_passes.empty:
        return df_live
        
    live_coords = df_live[df_live['x'].notna()].copy()
    matched_rows = []
    
    for _, live_row in live_coords.iterrows():
        team = live_row.get('team')
        lx = live_row['x']
        ly = live_row['y']
        ls_x = live_row['start_x'] if pd.notna(live_row.get('start_x')) else lx
        ls_y = live_row['start_y'] if pd.notna(live_row.get('start_y')) else ly
        
        team_passes = df_passes[df_passes['team'] == team].copy()
        if team_passes.empty:
            continue
            
        dist_start_end = np.sqrt(
            (team_passes['start_x'] - ls_x)**2 + (team_passes['start_y'] - ls_y)**2 +
            (team_passes['end_x'] - lx)**2 + (team_passes['end_y'] - ly)**2
        )
        
        best_idx = dist_start_end.idxmin()
        min_dist = dist_start_end[best_idx]
        best_row = team_passes.loc[best_idx]
        
        res = live_row.to_dict()
        if min_dist <= max_coord_distance:
            res['player_id'] = best_row['player_id']
            res['player_name'] = best_row['player_name']
            res['action_type'] = best_row['event_action_type']
            res['category'] = best_row['event_category']
            res['outcome'] = best_row['outcome']
            res['keypass'] = best_row['keypass']
            res['pass_start_x'] = best_row['start_x']
            res['pass_start_y'] = best_row['start_y']
            res['pass_end_x'] = best_row['end_x']
            res['pass_end_y'] = best_row['end_y']
            res['match_coord_dist'] = round(min_dist, 2)
        else:
            res['player_id'] = None
            res['player_name'] = None
            res['action_type'] = None
            res['category'] = None
            res['outcome'] = None
            res['keypass'] = None
            res['pass_start_x'] = None
            res['pass_start_y'] = None
            res['pass_end_x'] = None
            res['pass_end_y'] = None
            res['match_coord_dist'] = round(min_dist, 2)
            
        matched_rows.append(res)
        
    return pd.DataFrame(matched_rows)
