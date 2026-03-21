import sys
import pandas as pd
import datetime
import os

# Create project dir if script is moved
pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
os.makedirs(pipeline_dir, exist_ok=True)

# Add modules to path so tx_module can be imported
sys.path.append(r"E:\Data\TSDP\modules")
try:
    from tx_module import get_league_odds
except ImportError as e:
    print(f"Error importing tx_module: {e}. Make sure the path E:\\Data\\TSDP\\modules is correct.")
    sys.exit(1)

# NBA Team Mapping Dictionary
# Tippmixpro name -> NBA Abbreviation (feel free to add missing ones after the first run!)
TEAM_MAPPING = {
    'Atlanta Hawks': 'ATL', 'Atlanta': 'ATL',
    'Boston Celtics': 'BOS', 'Boston': 'BOS',
    'Brooklyn Nets': 'BKN', 'Brooklyn': 'BKN',
    'Charlotte Hornets': 'CHA', 'Charlotte': 'CHA',
    'Chicago Bulls': 'CHI', 'Chicago': 'CHI',
    'Cleveland Cavaliers': 'CLE', 'Cleveland': 'CLE',
    'Dallas Mavericks': 'DAL', 'Dallas': 'DAL',
    'Denver Nuggets': 'DEN', 'Denver': 'DEN',
    'Detroit Pistons': 'DET', 'Detroit': 'DET',
    'Golden State Warriors': 'GSW', 'Golden State': 'GSW', 'G. State Warriors': 'GSW', 'G. State': 'GSW',
    'Houston Rockets': 'HOU', 'Houston': 'HOU',
    'Indiana Pacers': 'IND', 'Indiana': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'L.A. Clippers': 'LAC',
    'LA Lakers': 'LAL', 'Los Angeles Lakers': 'LAL', 'L.A. Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Memphis': 'MEM',
    'Miami Heat': 'MIA', 'Miami': 'MIA',
    'Milwaukee Bucks': 'MIL', 'Milwaukee': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'Minnesota': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New Orleans': 'NOP',
    'New York Knicks': 'NYK', 'New York': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Oklahoma City': 'OKC', 'Oklahoma': 'OKC',
    'Orlando Magic': 'ORL', 'Orlando': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Philadelphia': 'PHI', 'Phila 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Phoenix': 'PHX',
    'Portland Trail Blazers': 'POR', 'Portland': 'POR',
    'Sacramento Kings': 'SAC', 'Sacramento': 'SAC',
    'San Antonio Spurs': 'SAS', 'San Antonio': 'SAS',
    'Toronto Raptors': 'TOR', 'Toronto': 'TOR',
    'Utah Jazz': 'UTA', 'Utah': 'UTA',
    'Washington Wizards': 'WAS', 'Washington': 'WAS'
}

def map_team_name(scraped_name):
    # Exact Match
    if scraped_name in TEAM_MAPPING:
        return TEAM_MAPPING[scraped_name]
    
    # Substring Match (Case-insensitive)
    for key, abbr in TEAM_MAPPING.items():
        if key.lower() in scraped_name.lower() or scraped_name.lower() in key.lower():
            return abbr
            
    return scraped_name  # Return original if entirely missing

def get_live_nba_odds(url="https://www.tippmixpro.hu/hu/fogadas/i/bajnoksag-lokacio/kosarlabda/8/usa/229/nba-2025-2026/274663790763708416"):
    """
    Scrapes the specified URL, aligns data to standard NBA IDs, and prints manual verification.
    Note: The URL might need to be adjusted exactly to Tippmix NBA's landing page!
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Lépés 1: Fetching live NBA Odds from {url}")
    df_matches = get_league_odds(url, headless=True)
    
    if df_matches is None or df_matches.empty:
        print("\n❌ HIBA: Nem talált meccseket a scraper! (Rossz URL, vagy üres a kínálat jelenleg?)")
        return pd.DataFrame()
        
    print(f"✅ Siker: {len(df_matches)} mérkőzés lekaparva. Csapatnevek fordítása a modell nyelvére...")
    
    df_matches['Home_Abbr'] = df_matches['home_team'].apply(map_team_name)
    df_matches['Away_Abbr'] = df_matches['away_team'].apply(map_team_name)
    
    # Validation
    unmapped = df_matches[(df_matches['Home_Abbr'] == df_matches['home_team']) | 
                          (df_matches['Away_Abbr'] == df_matches['away_team'])]
    
    if not unmapped.empty:
        print("\n⚠️ FIGYELEM: Vannak ismeretlen csapatnevek! Bővíteni kell a TEAM_MAPPING szótárat a kódban:")
        for _, row in unmapped.iterrows():
            if row['Home_Abbr'] == row['home_team']:
                 print(f"   [!] Ismeretlen Hazai: '{row['home_team']}'")
            if row['Away_Abbr'] == row['away_team']:
                 print(f"   [!] Ismeretlen Vendég: '{row['away_team']}'")
                 
    print("\n=======================================================")
    print("--- STAGE 1 OUTPUT: MANUAL REALITY CHECK (EYE TEST) ---")
    print("=======================================================")
    view_df = df_matches[['date', 'away_team', 'Away_Abbr', 'home_team', 'Home_Abbr', 'away_odds', 'home_odds']].copy()
    print(view_df.to_string(index=False))
    print("\n=======================================================")
    
    # Save the staging data so Stage 2 can easily read it during testing
    stage1_path = os.path.join(pipeline_dir, "staging_1_odds.csv")
    df_matches.to_csv(stage1_path, index=False)
    print(f"Staging Data elmentve as: {stage1_path}")
    
    return df_matches

if __name__ == "__main__":
    # Use the default URL defined in the function signature
    get_live_nba_odds()
