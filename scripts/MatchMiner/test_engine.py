import pandas as pd
from data_loader import load_team_data, load_player_data
from insight_engine import analyze_game
from narrative_formatter import format_team_insight, format_player_insight

def test():
    team_path = "data/team0630_1752.csv"
    player_path = "data/player0630_1752.csv"
    
    print("Loading data...")
    team_df = load_team_data(team_path)
    player_df = load_player_data(player_path)
    
    print(f"Team rows: {len(team_df)}, Player rows: {len(player_df)}")
    
    # Let's get unique gameIds
    games = team_df['gameId'].unique()
    print(f"Total unique games: {len(games)}")
    
    if len(games) > 0:
        target_game = games[0]
        print(f"\nAnalyzing first game: {target_game}")
        print(team_df[team_df['gameId'] == target_game][['Date', 'game', 'leagueName']])
        
        results = analyze_game(team_df, player_df, target_game, min_player_minutes=15, min_z_score=1.5)
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nFound {len(results['team_insights'])} team insights and {len(results['player_insights'])} player insights.")
            
            print("\n--- TOP TEAM INSIGHTS ---")
            for ins in results['team_insights'][:3]:
                print(f"Score: {ins['score']} | Z: {ins['z_score']:.2f} | Record: {ins['is_record']}")
                print(format_team_insight(ins))
                
            print("\n--- TOP PLAYER INSIGHTS ---")
            for ins in results['player_insights'][:3]:
                print(f"Score: {ins['score']} | Z-Pos: {ins['z_score_pos']:.2f} | Record: {ins['is_record']}")
                print(format_player_insight(ins))

if __name__ == "__main__":
    test()
