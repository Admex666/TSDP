def clubelo_fixtures_to_df():
    import requests
    import pandas as pd
    import io  # Add this line to import the 'io' module

    # URL of the CSV file
    url = "http://api.clubelo.com/Fixtures"

    # Sending a GET request to download the CSV file
    response = requests.get(url)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Saving the content of the response (CSV data)
        csv_data = response.content

        # Using pandas to read the CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
        return df
    else:
        print("Failed to download the CSV file.")

def tippmix_api_to_dataframe():
    import requests
    import pandas as pd
    # URL of the API endpoint
    url = "https://api.tippmix.hu/event"

    # Send a GET request to the API endpoint
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the JSON data from the response
        data = response.json()

        # Check if the data is a dictionary with the key "data"
        if isinstance(data, dict) and "data" in data:
            # Extract the list of events from the data
            events = data["data"]

            # Initialize lists to store data
            dates = []
            home_participants = []
            away_participants = []
            home_fixed_odds = []
            draw_fixed_odds = []
            away_fixed_odds = []

            # Loop through the events to find matches where the sportName is "Labdarúgás"
            for event in events:
                if event.get("sportName") == "Labdarúgás":
                    # Extract event data
                    event_date = str(event.get("eventDate"))
                    event_date = event_date.split('T')[0]
                    home_participant = event.get("eventParticipants")[0].get("participantName")
                    away_participant = event.get("eventParticipants")[1].get("participantName")
                    
                    # Extract markets for the event
                    markets = event.get("markets", [])
                    
                    # Loop through the markets to find the one with marketName "1X2"
                    for market in markets:
                        if market.get("marketName") == "1X2":
                            # Extract odds for each outcome
                            outcomes = market.get("outcomes", [])
                            home_odds = outcomes[0].get("fixedOdds")
                            draw_odds = outcomes[1].get("fixedOdds")
                            away_odds = outcomes[2].get("fixedOdds")
                            
                            # Append data to lists
                            dates.append(event_date)
                            home_participants.append(home_participant)
                            away_participants.append(away_participant)
                            home_fixed_odds.append(home_odds)
                            draw_fixed_odds.append(draw_odds)
                            away_fixed_odds.append(away_odds)

            # Create a DataFrame from the lists
            df = pd.DataFrame({
                "Date": dates,
                "Home": home_participants,
                "Away": away_participants,
                "home_odds": home_fixed_odds,
                "draw_odds": draw_fixed_odds,
                "away_odds": away_fixed_odds
            })

            return df

        else:
            print("Data format is not as expected")
            return None
    else:
        print("Failed to retrieve data from the API")
        return None
    
def scrape_odds_to_excel_horizontal(date):
    
    #importing libraries
    from bs4 import BeautifulSoup
    import requests
    import json
    import pandas as pd
    from fuzzywuzzy import fuzz, process
    import time
    st = time.time()
    
    df_tippmix = tippmix_api_to_dataframe()
    
    df_fixtures = clubelo_fixtures_to_df()
    df_fixtures['home_odds'] = df_fixtures['R:1-0'] + df_fixtures['R:2-0'] + df_fixtures['R:3-0']+ df_fixtures['R:4-0'] + df_fixtures['R:5-0'] + df_fixtures['R:6-0']+ df_fixtures['R:2-1'] + df_fixtures['R:3-1'] + df_fixtures['R:4-1'] + df_fixtures['R:5-1'] + df_fixtures['R:3-2'] + df_fixtures['R:4-2']
    df_fixtures['draw_odds'] = df_fixtures['R:0-0'] + df_fixtures['R:1-1'] + df_fixtures['R:2-2']+ df_fixtures['R:3-3']
    df_fixtures['away_odds'] = df_fixtures['R:0-1'] + df_fixtures['R:0-2'] + df_fixtures['R:0-3']+ df_fixtures['R:0-4'] + df_fixtures['R:0-5'] + df_fixtures['R:0-6']+ df_fixtures['R:1-2'] + df_fixtures['R:1-3'] + df_fixtures['R:1-4'] + df_fixtures['R:1-5'] + df_fixtures['R:2-3'] + df_fixtures['R:2-4']
    #making the sum of odds 1
    df_fixtures['sum_odds'] = df_fixtures['home_odds'] + df_fixtures['draw_odds'] +df_fixtures['away_odds']
    df_fixtures['home_odds'] = df_fixtures['home_odds']/df_fixtures['sum_odds']
    df_fixtures['draw_odds'] = df_fixtures['draw_odds']/df_fixtures['sum_odds']
    df_fixtures['away_odds'] = df_fixtures['away_odds']/df_fixtures['sum_odds']
        
    df_elo_odds = df_fixtures[['Date','Country', 'Home', 'Away', 'home_odds', 'draw_odds', 'away_odds']].copy()
    
    merge_dfs = pd.DataFrame(columns=['country', 'home_team', 'away_team', 'home_odds', 'draw_odds', 'away_odds', 'elo_home_odds', 'elo_draw_odds', 'elo_away_odds', 'home_to_bet', 'draw_to_bet', 'away_to_bet'])

    # Extracting home team, away team, and odds
    for index, row in df_tippmix.iterrows():
        if date == row['Date']:
            home_team = row['Home']
            away_team = row['Away']
            home_odds = row['home_odds']
            draw_odds = row['draw_odds']
            away_odds = row['away_odds']

            #checking similarity of team names
            def team_names_check():
                H_max_ratio = 0
                A_max_ratio = 0
                for H_team_name in df_elo_odds['Home']:
                    team_column = 'Home'
                    H_ratio = fuzz.partial_ratio(H_team_name.upper(), home_team.upper())
                    if H_ratio > H_max_ratio:
                        H_max_ratio = H_ratio
                        H_best_team = H_team_name
                for A_team_name in df_elo_odds['Away']:
                    A_ratio = fuzz.partial_ratio(A_team_name.upper(), away_team.upper())
                    if A_ratio > A_max_ratio:
                        A_max_ratio = A_ratio
                        A_best_team = A_team_name
                return [H_best_team, A_best_team, H_max_ratio, A_max_ratio]

            H_best_team = team_names_check()[0]
            A_best_team = team_names_check()[1]
            H_max_ratio = team_names_check()[2]
            A_max_ratio = team_names_check()[3]
            
            #only run if row indices are the equal (same match)
            home_indices = df_elo_odds[df_elo_odds['Home'] == H_best_team].index
            away_indices = df_elo_odds[df_elo_odds['Away'] == A_best_team].index

            if not home_indices.empty and not away_indices.empty and home_indices[0] == away_indices[0]:
                if (H_max_ratio > 70 and A_max_ratio > 45) or (H_max_ratio > 45 and A_max_ratio > 70):
                    country = str(df_elo_odds[df_elo_odds['Home'] == H_best_team]['Country'])
                    country = country.split()[1].strip()

                    elo_home_probs = str(df_elo_odds[df_elo_odds['Home'] == H_best_team]['home_odds'])
                    elo_home_probs = float(elo_home_probs.split()[1])
                    elo_draw_probs = str(df_elo_odds[df_elo_odds['Home'] == H_best_team]['draw_odds'])
                    elo_draw_probs = float(elo_draw_probs.split()[1])
                    elo_away_probs = str(df_elo_odds[df_elo_odds['Home'] == H_best_team]['away_odds'])
                    elo_away_probs = float(elo_away_probs.split()[1])

                    elo_home_odds = round(float(1/elo_home_probs), 2)
                    elo_draw_odds = round(float(1/elo_draw_probs), 2)
                    elo_away_odds = round(float(1/elo_away_probs), 2)

                    if home_odds >= elo_home_odds:
                        home_to_bet = 1/(home_odds-1)
                    else:
                        home_to_bet = 0

                    if draw_odds >= elo_draw_odds:
                        draw_to_bet = 1/(draw_odds-1)
                    else:
                        draw_to_bet = 0

                    if away_odds >= elo_away_odds:
                        away_to_bet = 1/(away_odds-1)
                    else:
                        away_to_bet = 0

                    df = pd.DataFrame(columns = ['country', 'home_team', 'away_team', 'home_odds', 'draw_odds', 'away_odds', 'elo_home_odds', 'elo_draw_odds', 'elo_away_odds', 'home_to_bet', 'draw_to_bet', 'away_to_bet'])

                    match_row = {'country': country, 'home_team': home_team, 'away_team': away_team, 'home_odds': home_odds, 'draw_odds': draw_odds, 'away_odds': away_odds, 'elo_home_odds': elo_home_odds, 'elo_draw_odds': elo_draw_odds, 'elo_away_odds': elo_away_odds, 'home_to_bet': home_to_bet, 'draw_to_bet': draw_to_bet, 'away_to_bet': away_to_bet}
                    df = df._append(match_row, ignore_index=True)

                    merge_dfs = pd.concat([merge_dfs, df], ignore_index=True)
    
    path = r'C:\Users\Adam\Downloads\scrape_odds_horizontal.xlsx'
    merge_dfs.to_excel(path, sheet_name = date, index=False)
    
    if merge_dfs.shape[0] == 0:
        print("No matches were found.")
    else:
        print(merge_dfs.shape[0], "matches were found, and added to the Excel file.")
    
    et = time.time()
    elapsed = round(et-st, 2)
    print('')
    print('Elapsed time:', elapsed, 'seconds')