import pandas as pd
import os

def main():
    data_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\data"
    output_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321"
    
    print("Loading datasets...")
    # Load data
    ml_ready = pd.read_csv(os.path.join(data_dir, "ml_ready_2024_25.csv"))
    games = pd.read_csv(os.path.join(data_dir, "games_2024_25.csv"))
    odds = pd.read_csv(os.path.join(data_dir, "game_odds_2024_25.csv"))
    
    # ml_ready game_id is usually formatted with leading zeros.
    # make sure they are strings for merging
    ml_ready['game_id'] = ml_ready['game_id'].astype(str).str.zfill(10)
    games['GAME_ID'] = games['GAME_ID'].astype(str).str.zfill(10)
    odds['GAME_ID'] = odds['GAME_ID'].astype(str).str.zfill(10)
    
    # Extract Target from games
    # Find home teams: 'MATCHUP' contains ' vs. '
    home_games = games[games['MATCHUP'].str.contains(' vs. ', na=False)].copy()
    home_games['home_win'] = (home_games['WL'] == 'W').astype(int)
    home_games['home_margin'] = home_games['PLUS_MINUS']
    target_df = home_games[['GAME_ID', 'GAME_DATE', 'home_win', 'home_margin']]
    
    # Merge ml_ready with target and date
    df = pd.merge(ml_ready, target_df, left_on='game_id', right_on='GAME_ID', how='inner')
    print(f"Games matched with results: {len(df)}")
    
    # Merge with odds
    df = pd.merge(df, odds[['GAME_ID', 'odds_home', 'odds_away']], on='GAME_ID', how='inner')
    print(f"Games matched with odds: {len(df)}")
    
    # Drop rows with missing values that are essential
    df.dropna(subset=['home_win', 'home_margin', 'odds_home', 'odds_away'], inplace=True)
    df.fillna(0, inplace=True) # Fill numerical NAs with 0 where necessary
    
    # Sort chronologically
    df.sort_values('GAME_DATE', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Final dataset size: {len(df)}")
    print(f"Home Win rate in dataset: {df['home_win'].mean():.2f}")
    
    # Save the preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()
