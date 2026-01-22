import pandas as pd
import requests
import os
import time

def download_data(leagues, seasons):
    """
    Downloads football data CSVs from football-data.co.uk.
    leagues: list of league codes (e.g., ['E0', 'E1'])
    seasons: list of season codes (e.g., ['2425', '2324'])
    """
    base_url = "https://www.football-data.co.uk/mmz4281/"
    os.makedirs('data', exist_ok=True)
    
    all_data = []
    
    for season in seasons:
        for league in leagues:
            url = f"{base_url}{season}/{league}.csv"
            filename = f"data/{league}_{season}.csv"
            
            if not os.path.exists(filename):
                print(f"Downloading {url}...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    time.sleep(1) # Be nice to the server
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    continue
            
            try:
                # Some CSVs might have different encodings or structures
                df = pd.read_csv(filename, on_bad_lines='skip')
                # Keep relevant columns
                cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
                # Filter columns that exist
                existing_cols = [c for c in cols if c in df.columns]
                df = df[existing_cols]
                df['Season'] = season
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    if not all_data:
        return pd.DataFrame()
        
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Process Date to datetime
    # The format can vary: dd/mm/yy or dd/mm/yyyy
    def parse_date(date_str):
        if not isinstance(date_str, str): return pd.NaT
        for fmt in ('%d/%m/%Y', '%d/%m/%y'):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return pd.to_datetime(date_str, errors='coerce')

    master_df['Date'] = master_df['Date'].apply(parse_date)
    master_df = master_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    master_df = master_df.sort_values('Date').reset_index(drop=True)
    
    return master_df

if __name__ == "__main__":
    # Example usage: Premier League for last 3 seasons
    leagues = ['E0'] # Premier League
    seasons = ['2425', '2324', '2223']
    df = download_data(leagues, seasons)
    print(f"Loaded {len(df)} matches.")
    df.to_csv('data/master_football_data.csv', index=False)
