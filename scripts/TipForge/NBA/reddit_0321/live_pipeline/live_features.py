import sys
import os
import pandas as pd
import datetime

pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
# Add TipForge\NBA root to path so we can import the old run_predictions.py
sys.path.append(r"E:\Data\TSDP\scripts\TipForge\NBA")

try:
    from run_predictions import get_upcoming_games, create_pregame_features, calculate_differential_features
    from nba_api_module import TEAM_ABBREVIATIONS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure run_predictions.py and nba_api_module.py specify their paths correctly.")
    sys.exit(1)

def match_games(odds_df, nba_games_df):
    # Reverse mapping from Full Name to 3-letter Abbr
    nba_games_df['Home_Abbr'] = nba_games_df['home_team'].map(TEAM_ABBREVIATIONS)
    nba_games_df['Away_Abbr'] = nba_games_df['away_team'].map(TEAM_ABBREVIATIONS)

    # In case of date timezone mismatches, the safest join is by Home_Abbr and Away_Abbr 
    # since teams never play each other on consecutive days in the NBA regular season.
    merged = pd.merge(odds_df, nba_games_df, on=['Home_Abbr', 'Away_Abbr'], how='inner', suffixes=('_tippmix', '_nba'))
    return merged

def generate_live_features():
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Lépés 2: Feature Engineering & API lekérdezések")
    
    odds_path = os.path.join(pipeline_dir, "staging_1_odds.csv")
    if not os.path.exists(odds_path):
        print("❌ Hiba: Nincs Stage 1 fájl (staging_1_odds.csv). Futtasd elobb a live_odds.py-t!")
        return
        
    odds_df = pd.read_csv(odds_path)
    print(f"Bemenet: {len(odds_df)} meccs a Tippmixről.")
    
    print("Fetching scheduled NBA games from API (nba_api)... this might take a minute.")
    nba_schedule = get_upcoming_games(days_ahead=3)
    
    matched_games = match_games(odds_df, nba_schedule)
    print(f"\nSikeresen párosítva {len(matched_games)} mérkőzés a hivatalos NBA naptárral!")
    
    if len(matched_games) == 0:
        print("❌ Nincs egyezés a Tippmix meccsek és az NBA naptár között. Lehet, hogy ma nincsenek meccsek?")
        return
        
    all_features = []
    
    for idx, row in matched_games.iterrows():
        print(f"\n--- Feature generálás: {row['Away_Abbr']} @ {row['Home_Abbr']} (NBA Dátum: {row['game_date']}) ---")
        try:
            features = create_pregame_features(
                row['game_id'], '2025-26', row['game_date'],
                [row['home_team_id'], row['away_team_id']]
            )
            
            if features is None:
                print(f"⚠️ Nem sikerült generálni: {row['Home_Abbr']}")
                continue
                
            diff_features = calculate_differential_features(features)
            
            # Combine all previous values into the vector for downstream
            diff_features['game_id'] = row['game_id']
            diff_features['game_date'] = row['game_date']
            diff_features['home_team'] = row['home_team_nba']
            diff_features['away_team'] = row['away_team_nba']
            diff_features['Home_Abbr'] = row['Home_Abbr']
            diff_features['Away_Abbr'] = row['Away_Abbr']
            diff_features['home_odds'] = row['home_odds']
            diff_features['away_odds'] = row['away_odds']
            
            # Combine the original non-diff features for the Eye Test print
            diff_features.update(features)
            
            all_features.append(diff_features)
            
            # Eye-Test Printing
            print(f"  > Rest Days Diff: {diff_features['rest_days_diff']} (Home: {features['home_rest_days']}, Away: {features['away_rest_days']})")
            print(f"  > B2B Advantage:  {diff_features['b2b_advantage']} (Home B2B: {features['home_is_back_to_back']}, Away B2B: {features['away_is_back_to_back']})")
            print(f"  > NET Rtg Diff:   {diff_features['NET_rtg_diff']:.2f} (Home: {features['home_NET_rtg']:.2f}, Away: {features['away_NET_rtg']:.2f})")
            print(f"  > PACE Diff:      {diff_features['PACE_diff']:.2f} (Home: {features['home_PACE']:.2f}, Away: {features['away_PACE']:.2f})")
            
        except Exception as e:
            print(f"Hiba {row['Home_Abbr']} feldolgozásakor: {e}")
            
    features_df = pd.DataFrame(all_features)
    
    out_path = os.path.join(pipeline_dir, "staging_2_features.csv")
    features_df.to_csv(out_path, index=False)
    
    print("\n=======================================================")
    print("--- STAGE 2 OUTPUT: MANUAL REALITY CHECK (EYE TEST) ---")
    print("=======================================================")
    print(f"Összesen legenerált meccsek: {len(features_df)}")
    if not features_df.empty:
        first_game = features_df.iloc[0]
        print(f"Példa - Első meccs: {first_game['away_team']} @ {first_game['home_team']}")
        print(f"H-Form10: {first_game.get('home_recent_form10', 'N/A')}, A-Form10: {first_game.get('away_recent_form10', 'N/A')}")
        print(f"H-Injuries: {first_game.get('home_injury_count', 'N/A')}, A-Injuries: {first_game.get('away_injury_count', 'N/A')}")
    print("=======================================================")
    print(f"Staging Data elmentve: {out_path}")

if __name__ == "__main__":
    generate_live_features()
