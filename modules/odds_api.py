def odds_api(REGIONS = "eu", MARKETS = "h2h,totals", ODDS_FORMAT = "decimal"):
    import requests
    import pandas as pd
    import json
    from dotenv import load_dotenv
    import os

    load_dotenv()

    sport_key_list = ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga',
                'soccer_italy_serie_a', 'soccer_france_ligue_one', 
                'soccer_belgium_first_div','soccer_portugal_primeira_liga', 
                'soccer_netherlands_eredivisie',
                #'soccer_usa_mls','soccer_austria_bundesliga', 'soccer_uefa_champs_league','soccer_uefa_europa_league', 'soccer_uefa_europa_conference_league'
                ]
    data = []

    API_KEY = os.getenv("ODDSAPI_KEY")

    for SPORT in sport_key_list:
        # API kérés elküldése
        url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
        params = {
            "apiKey": API_KEY,
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": ODDS_FORMAT
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Hiba történt: {response.status_code}")
        
        
        data_league = response.json()

        data = [*data, *data_league]

    #%% DataFrame összeállítása
    rows = []

    for match in data:
        # Alap adatok kinyerése
        home_team = match["home_team"]
        away_team = match["away_team"]
        commence_time = match["commence_time"]
        
        # Fogadóirodák feldolgozása
        for bookmaker in match["bookmakers"]:
            bookmaker_name = bookmaker["title"]
            
            # Oddszok inicializálása
            odds = {
                "home": None,
                "draw": None,
                "away": None,
                "over_2.5": None,
                "under_2.5": None
            }
            
            # Piacok feldolgozása
            for market in bookmaker["markets"]:
                # H2H piac (1-X-2)
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_team:
                            odds["home"] = outcome["price"]
                        elif outcome["name"] == away_team:
                            odds["away"] = outcome["price"]
                        elif outcome["name"] == "Draw":
                            odds["draw"] = outcome["price"]
                
                # Totals piac (Over/Under)
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["point"] == 2.5:  # Csak a 2.5 gólhatárt nézzük
                            if outcome["name"] == "Over":
                                odds["over_2.5"] = outcome["price"]
                            elif outcome["name"] == "Under":
                                odds["under_2.5"] = outcome["price"]
            
            # Sor hozzáadása
            rows.append({
                "Home": home_team,
                "Away": away_team,
                "Date": commence_time,
                "bookmaker": bookmaker_name,
                "H_odds": odds["home"],
                "D_odds": odds["draw"],
                "A_odds": odds["away"],
                "Over_odds": odds["over_2.5"],
                "Under_odds": odds["under_2.5"]
            })

    # DataFrame létrehozása
    df = pd.DataFrame(rows)
    df.Date = pd.to_datetime(df.Date).dt.tz_localize(None)

    # Eredmények megjelenítése
    print(f"{len(df)} odds found:")
    print(df.head())

    print('Remaining requests', response.headers['x-requests-remaining'])
    print('Used requests', response.headers['x-requests-used'])

    return df