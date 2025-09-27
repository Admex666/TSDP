# debug_odds_comparison.py
import pandas as pd
from mlb_data_loader import get_upcoming_mlb_games
from mlb_tippmix_api import get_mlb_tippmix_data
from mlb_config import API_TO_TIPPMIX

def debug_odds_differences():
    """Debug script to compare odds between implementations"""
    
    print("=== DEBUGGING ODDS DIFFERENCES ===\n")
    
    # Get games and odds data
    upcoming_games = get_upcoming_mlb_games(1)
    tippmix_data = get_mlb_tippmix_data(1)
    
    print(f"Found {len(upcoming_games)} upcoming games")
    print(f"Found {len(tippmix_data)} tippmix odds")
    
    print("\n--- TIPPMIX DATA STRUCTURE ---")
    if not tippmix_data.empty:
        print("Columns:", tippmix_data.columns.tolist())
        print("First row:", tippmix_data.iloc[0].to_dict())
        
        print("\n--- ALL AVAILABLE MATCHES ---")
        for i, odds in tippmix_data.iterrows():
            print(f"  {odds['Home']} vs {odds['Away']}")
    
    print("\n--- UPCOMING GAMES MATCHING ---")
    for _, game in upcoming_games.iterrows():
        home_team = game['home_team_name']
        away_team = game['away_team_name']
        
        print(f"\nGame: {home_team} vs {away_team}")
        
        # Check mapping
        home_tippmix = API_TO_TIPPMIX.get(home_team, home_team)
        away_tippmix = API_TO_TIPPMIX.get(away_team, away_team)
        
        print(f"  Mapped to: {home_tippmix} vs {away_tippmix}")
        
        # Try to find odds
        odds_row = None
        for _, odds in tippmix_data.iterrows():
            if (odds['Home'].strip().lower() == home_tippmix.strip().lower() and 
                odds['Away'].strip().lower() == away_tippmix.strip().lower()):
                odds_row = odds
                break
        
        if odds_row is not None:
            # Check which odds fields exist
            odds_fields = []
            for field in ['Home_odds', 'Away_odds', 'H_odds', 'A_odds']:
                if field in odds_row:
                    odds_fields.append(f"{field}: {odds_row[field]}")
            
            print(f"  ✅ FOUND ODDS: {', '.join(odds_fields)}")
        else:
            print(f"  ❌ NO ODDS FOUND")
            
            # Try partial matching for debugging
            print("  Partial matches:")
            for _, odds in tippmix_data.iterrows():
                home_match = home_tippmix.lower() in odds['Home'].lower() or odds['Home'].lower() in home_tippmix.lower()
                away_match = away_tippmix.lower() in odds['Away'].lower() or odds['Away'].lower() in away_tippmix.lower()
                
                if home_match or away_match:
                    print(f"    Partial: {odds['Home']} vs {odds['Away']}")

def compare_team_mappings():
    """Compare team mappings between implementations"""
    
    print("\n=== TEAM MAPPING COMPARISON ===\n")
    
    # From real_predictor.ipynb
    api_to_tippmix_notebook = {
        "Tampa Bay Rays": "Tampa Bay",
        "Baltimore Orioles": "Baltimore",
        "Pittsburgh Pirates": "Pittsburgh",
        "Texas Rangers": "Texas",
        "Detroit Tigers": "Detroit",
        "New York Yankees": "NY Yankees",
        "Kansas City Royals": "Kansas City",
        "Chicago White Sox": "Chicago WS",
        "Arizona Diamondbacks": "Arizona",
        "Houston Astros": "Houston",
        "St. Louis Cardinals": "St. Louis",
        "Colorado Rockies": "Colorado",
        "Los Angeles Dodgers": "LA Dodgers",
        "Los Angeles Angels": "LA Angels",
        "Cincinnati Reds": "Cincinnati",
        "Chicago Cubs": "Chicago Cubs",
        "Toronto Blue Jays": "Toronto",
        "Washington Nationals": "Washington",
        "New York Mets": "NY Mets",
        "Miami Marlins": "Miami",
        "Boston Red Sox": "Boston",
        "Philadelphia Phillies": "Philadelphia",
        "Cleveland Guardians": "Cleveland",
        "Minnesota Twins": "Minnesota",
        "Atlanta Braves": "Atlanta",
        "Milwaukee Brewers": "Milwaukee",
        "San Diego Padres": "San Diego",
        "San Francisco Giants": "San Francisco",
        "Seattle Mariners": "Seattle",
        "Athletics": "Las Vegas",
    }
    
    print("Differences in team mappings:")
    for team in set(list(api_to_tippmix_notebook.keys()) + list(API_TO_TIPPMIX.keys())):
        notebook_mapping = api_to_tippmix_notebook.get(team, "NOT FOUND")
        config_mapping = API_TO_TIPPMIX.get(team, "NOT FOUND")
        
        if notebook_mapping != config_mapping:
            print(f"  {team}:")
            print(f"    Notebook: {notebook_mapping}")
            print(f"    Config:   {config_mapping}")

def fix_odds_columns():
    """Show how to fix the odds column naming issue"""
    
    print("\n=== FIXING ODDS COLUMNS ===\n")
    
    tippmix_data = get_mlb_tippmix_data(1)
    
    if not tippmix_data.empty:
        print("Current columns:", tippmix_data.columns.tolist())
        
        # Check what odds columns we have
        odds_columns = [col for col in tippmix_data.columns if 'odds' in col.lower()]
        print("Odds columns found:", odds_columns)
        
        # Suggest fix
        print("\nSuggested fix for mlb_tippmix_api.py:")
        print("Replace:")
        print("  match_odds['Home_odds'] = market['outcomes'][0]['fixedOdds']")
        print("  match_odds['Away_odds'] = market['outcomes'][1]['fixedOdds']")
        print("With:")
        print("  match_odds['H_odds'] = market['outcomes'][0]['fixedOdds']")
        print("  match_odds['A_odds'] = market['outcomes'][1]['fixedOdds']")

if __name__ == "__main__":
    debug_odds_differences()
    compare_team_mappings()
    fix_odds_columns()