import pandas as pd
from data_loader import load_team_data, load_player_data
from insight_engine import analyze_game

def test_all_games():
    team_path = "data/team0630_1752.csv"
    player_path = "data/player0630_1752.csv"
    
    team_df = load_team_data(team_path)
    player_df = load_player_data(player_path)
    
    # Sort games by date
    unique_games = team_df.sort_values('Date')['gameId'].unique()
    
    print(f"Total unique games: {len(unique_games)}")
    
    games_with_team_insights = 0
    for idx, g_id in enumerate(unique_games):
        # Analyze game
        res = analyze_game(team_df, player_df, g_id, min_player_minutes=15, min_z_score=1.5)
        team_count = len(res.get('team_insights', []))
        player_count = len(res.get('player_insights', []))
        
        if team_count > 0:
            games_with_team_insights += 1
            if games_with_team_insights <= 5:
                # Print sample
                game_info = team_df[team_df['gameId'] == g_id].iloc[0]
                print(f"Game {idx+1}: {game_info['game']} ({game_info['Date'].strftime('%Y-%m-%d')}) -> Team Insights: {team_count}, Player Insights: {player_count}")
                print(f"  Top Team Insight: {res['team_insights'][0]['team']} - {res['team_insights'][0]['metric']} (Val: {res['team_insights'][0]['value']}, Z: {res['team_insights'][0]['z_score']:.2f})")
                
    print(f"\nGames with at least one team insight: {games_with_team_insights} / {len(unique_games)}")

if __name__ == "__main__":
    test_all_games()
