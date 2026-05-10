import pandas as pd
import json
import time
import random
from typing import Dict, Any, Optional
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