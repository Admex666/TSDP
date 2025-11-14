import pandas as pd
import json
import tls_client
from datetime import datetime

def scrape_tennis_match(event_id):
    """
    Tenisz meccs adatainak scrapelése SofaScore-ról
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}"
    
    sess = tls_client.Session(client_identifier="chrome_118")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }
    
    resp = sess.get(url, headers=headers)
    
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error: {resp.status_code}")
        return None

def get_tennis_match_data(event_id):
    """
    Tenisz meccs adatainak feldolgozása predikcióhoz
    """
    match_data = scrape_tennis_match(event_id)
    
    if not match_data:
        return None
    
    try:
        # Alap meccs információk
        event = match_data['event']
        tournament = event.get('tournament', {})
        date_int = int(event.get('startTimestamp', 0))
        date = datetime.fromtimestamp(date_int).strftime('%Y-%m-%d %H:%M')
        date_str = str(date)
        
        # Játékos információk
        home_player = match_data['event']['homeTeam']
        away_player = match_data['event']['awayTeam']
        
        # Felület típusa
        surface = tournament.get('surface', 'Unknown')
        surface_mapping = {
            'Hard': 'Hard',
            'Clay': 'Clay', 
            'Grass': 'Grass',
            'Carpet': 'Hard'  # Szőnyeg szintén kemény pályának tekintjük
        }
        surface_type = surface_mapping.get(surface, 'Hard')
        
        # Rangsor információk
        home_rank = home_player.get('ranking', 999)
        away_rank = away_player.get('ranking', 999)

        if (home_rank == 999) or (away_rank == 999):
            return None
        
        # Odds információk
        odds_data = get_tennis_odds(event_id)
        
        # Head-to-head információk
        h2h_data = get_head_to_head(event_id)
        
        # Összesített adatok
        match_info = {
            'date': date_str,
            'player1_name': home_player['name'],
            'player2_name': away_player['name'],
            'player1_rank': home_rank,
            'player2_rank': away_rank,
            'surface': surface_type,
            'tournament': tournament.get('name', 'Unknown'),
            'date': event.get('startTimestamp'),
            'player1_odds': odds_data.get('player1_odds', 2.0),
            'player2_odds': odds_data.get('player2_odds', 1.8),
            'h2h_matches': h2h_data.get('total_matches', 0),
            'h2h_p1_wins': h2h_data.get('player1_wins', 0),
            'h2h_last_winner': h2h_data.get('last_winner', 0.5)
        }
        
        return match_info
        
    except Exception as e:
        print(f"Hiba az adatok feldolgozásakor: {str(e)}")
        return None

def get_tennis_odds(event_id):
    """
    Tenisz meccs odds-ainak lekérése - JAVÍTOTT
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/featured"
    
    sess = tls_client.Session(client_identifier="chrome_118")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://www.sofascore.com/tennis/match/{event_id}",
        "Origin": "https://www.sofascore.com",
    }
    
    resp = sess.get(url, headers=headers)
    
    if resp.status_code == 200:
        odds_data = resp.json()
        
        try:
            featured = odds_data.get('featured', {})
            full_time = featured.get('fullTime', {})
            
            if full_time and 'choices' in full_time:
                choices = full_time['choices']
                if len(choices) >= 2:
                    # Fractional odds konvertálása decimal odds-ra
                    def fractional_to_decimal(fractional):
                        try:
                            num, denom = fractional.split('/')
                            return int(num) / int(denom) + 1
                        except:
                            return 2.0  # Alapértelmezett
                    
                    player1_odds = fractional_to_decimal(choices[0]['fractionalValue'])
                    player2_odds = fractional_to_decimal(choices[1]['fractionalValue'])
                    
                    return {
                        'player1_odds': player1_odds,
                        'player2_odds': player2_odds
                    }
        except Exception as e:
            print(f"Hiba az odds feldolgozásakor: {str(e)}")
    
    # Alapértelmezett értékek, ha nem sikerül lekérni
    return {'player1_odds': 1.00, 'player2_odds': 1.00}


def get_head_to_head(event_id):
    """
    Head-to-head statisztikák lekérése - JAVÍTOTT
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/h2h"
    
    sess = tls_client.Session(client_identifier="chrome_118")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://www.sofascore.com/tennis/match/{event_id}",
        "Origin": "https://www.sofascore.com",
    }
    
    resp = sess.get(url, headers=headers)
    
    if resp.status_code == 200:
        h2h_data = resp.json()
        
        try:
            team_duel = h2h_data.get('teamDuel', {})
            home_wins = team_duel.get('homeWins', 0)
            away_wins = team_duel.get('awayWins', 0)
            total_matches = home_wins + away_wins
            
            # Utolsó győztes meghatározása
            if total_matches > 0:
                # Feltételezzük, hogy a home player az 1-es játékos
                last_winner = 1 if home_wins > away_wins else 0
            else:
                last_winner = 0.5
                
            return {
                'total_matches': total_matches,
                'player1_wins': home_wins,
                'last_winner': last_winner
            }
            
        except Exception as e:
            print(f"Hiba a H2H adatok feldolgozásakor: {str(e)}")
    
    # Alapértelmezett értékek, ha nem sikerül lekérni
    return {'total_matches': 0, 'player1_wins': 0, 'last_winner': 0.5}

def create_prediction_input(match_info):
    """
    Predikciós input létrehozása a match_info alapján
    """
    p1_odds = match_info['player1_odds']
    p2_odds = match_info['player2_odds']
    p1_rank = match_info['player1_rank']
    p2_rank = match_info['player2_rank']
    h2h_total = match_info['h2h_matches']
    h2h_p1_wins = match_info['h2h_p1_wins']
    h2h_last = match_info['h2h_last_winner']
    
    # Feature-ök számítása (ugyanaz, mint az eredeti notebookban)
    odds_diff = p2_odds - p1_odds
    implied_prob1 = 1 / p1_odds
    implied_prob2 = 1 / p2_odds
    rank_diff = p2_rank - p1_rank
    h2h_p1_winrate = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5
    
    prediction_data = pd.DataFrame({
        'Odds_Diff': [odds_diff],
        'Implied_Prob1': [implied_prob1], 
        'Implied_Prob2': [implied_prob2],
        'H2H_P1_WinRate': [h2h_p1_winrate],
        'H2H_LastWinnerP1': [h2h_last],
        'Rank_Diff': [rank_diff]
    })
    
    return prediction_data

# Get matches of a date
def get_date_matches(date, min_points=250):
    """
    Mai nap tenisz meccseinek gyűjtése ATP 250+ tornákról
    """
    from datetime import datetime
    
    # Mai dátum formázása
    url = f"https://www.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}"
    
    sess = tls_client.Session(client_identifier="chrome_118")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "Origin": "https://www.sofascore.com",
    }
    
    resp = sess.get(url, headers=headers)
    
    if resp.status_code == 200:
        data = resp.json()
        match_ids = []
        
        for event in data.get('events', []):
            # Csak ATP 250+ tornák és még nem kezdődött el
            tournament_points = event.get('tournament', {}).get('uniqueTournament', {}).get('tennisPoints', 0)
            
            if (event.get('status', {}).get('code') == 0 and 
                tournament_points >= min_points):
                
                match_ids.append(event['id'])
                print(f"✅ Meccs hozzáadva: {event['homeTeam']['name']} vs {event['awayTeam']['name']} (ID: {event['id']}, Pontok: {tournament_points})")
            else:
                status_code = event.get('status', {}).get('code')
                points = tournament_points
                print(f"❌ Meccs kihagyva (status: {status_code}, pontok: {points}): {event['homeTeam']['name']} vs {event['awayTeam']['name']}")
        
        return match_ids
    else:
        print(f"❌ Hiba a mai meccsek lekérésekor: {resp.status_code}")
        return []