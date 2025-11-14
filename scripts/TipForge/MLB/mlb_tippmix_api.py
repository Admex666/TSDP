# mlb_tippmix_api.py
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_mlb_tippmix_data(days=1):
    """
    MLB mérkőzések odds adatainak lekérése Tippmixről
    """
    url = 'https://api.tippmix.hu/event'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data is None:
            print("No data received from Tippmix")
            return pd.DataFrame()

        matches = data['data']
        today_date = datetime.today().date()
        span = timedelta(days=days)

        matches_filtered = []
        for match in matches:
            match_date_iso = datetime.fromisoformat(match['eventDate'])
            match_date = match_date_iso.date()
            
            # Filter for MLB games within the specified timespan
            filter_condition = (
                (match['sportId'] == 3) and  # Baseball sport ID
                (match.get('competitionName') == 'MLB') and 
                (match_date <= today_date + span)
            )
            
            if filter_condition:
                matches_filtered.append(match)

        # Find odds from filtered matches
        odds_data = []
        for match in matches_filtered:
            match_odds = {}
            match_date_iso = datetime.fromisoformat(match['eventDate'])
            match_odds['Date'] = datetime.combine(match_date_iso.date(), match_date_iso.time())
            match_odds['Home'] = match['eventParticipants'][0]['participantName']
            match_odds['Away'] = match['eventParticipants'][1]['participantName']
            match_odds['competition_name'] = match.get('competitionName', 'MLB')
            
            # Look for moneyline odds (winner market)
            found_odds = False
            for market in match['markets']:
                if market['marketName'] == 'A mérkőzés győztese':
                    if len(market['outcomes']) >= 2:
                        match_odds['H_odds'] = market['outcomes'][0]['fixedOdds']
                        match_odds['A_odds'] = market['outcomes'][1]['fixedOdds']
                        found_odds = True
                    break
            
            if found_odds:
                odds_data.append(match_odds)

        return pd.DataFrame(odds_data)
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP error occurred: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Other error occurred: {e}")
        return pd.DataFrame()
