"""
SofaScore scraper module integration
"""
import pandas as pd
import json


def scrape_sofascore(url):
    import tls_client
    import time
    import random

    # Válassz "client_profile"-t ami Chrome/Safari-szerű fingerprintet ad.
    sess = tls_client.Session(client_identifier="chrome_118")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }

    resp = sess.get(url, headers=headers)

    if resp.status_code == 200:
        data = resp.json()
        return data
    else:
        print(f"Error: {resp.status_code}")
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
