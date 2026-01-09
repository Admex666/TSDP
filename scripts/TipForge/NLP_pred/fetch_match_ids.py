import soccerdata as sd
import pandas as pd

def get_match_ids(league="ENG-Premier League", season="2324"):
    """
    Fetches all match IDs for a given league and season using soccerdata.
    
    Args:
        league (str): The league ID (e.g., 'ENG-Premier League', 'GER-Bundesliga').
        season (str/int): The season (e.g., '2324' or 2023).
        
    Returns:
        pd.Series: A series containing unique match IDs.
    """
    print(f"Fetching schedule for {league}, season {season}...")
    
    # Initialize FBref reader
    fbref = sd.FBref(leagues=league, seasons=season)
    
    # read_schedule() is the most direct way to get all matches for the selected league/season.
    # It already includes a 'game_id' column which is the same as the ID in the match_report URL.
    schedule = fbref.read_schedule()
    
    if 'game_id' in schedule.columns:
        # Get unique game IDs, dropping any missing values (N/A for future matches)
        match_ids = schedule['game_id'].dropna().unique()
        return pd.Series(match_ids)
    else:
        # Fallback if game_id is missing (shouldn't happen with recent soccerdata)
        print("Warning: 'game_id' column not found. Extracting from 'match_report'...")
        match_ids = (
            schedule['match_report']
            .str.split('/')
            .str[3]
            .dropna()
            .unique()
        )
        return pd.Series(match_ids)

if __name__ == "__main__":
    # Example usage for Premier League 23/24
    league_id = "ENG-Premier League"
    season_id = "2324"
    
    ids = get_match_ids(league_id, season_id)
    
    print(f"\nFound {len(ids)} matches.")
    print(f"First 10 Match IDs (total: {len(ids)}):")
    print(ids.head(10).tolist())
    
    # Save to CSV if needed
    # ids.to_csv(f"match_ids_{league_id}_{season_id}.csv", index=False, header=['match_id'])
    # print(f"\nSaved to match_ids_{league_id}_{season_id}.csv")
