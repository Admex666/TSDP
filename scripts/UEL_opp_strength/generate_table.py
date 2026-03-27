import pandas as pd
import os

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    matches_file = os.path.join(directory, "uel_2025_26_rounds_1-8.csv")
    
    if not os.path.exists(matches_file):
        print(f"Error: {matches_file} not found.")
        return
        
    df = pd.read_csv(matches_file)
    
    # Dictionary to store team stats
    # Key: team_id, Value: dict of stats
    teams = {}
    
    def get_team_stats(team_id, team_name):
        if team_id not in teams:
            teams[team_id] = {
                'team_id': team_id,
                'team_name': team_name,
                'played': 0,
                'won': 0,
                'drawn': 0,
                'lost': 0,
                'gf': 0,
                'ga': 0,
                'points': 0
            }
        return teams[team_id]

    for _, row in df.iterrows():
        home_id = row['homeTeam_id']
        home_name = row['homeTeam_name']
        away_id = row['awayTeam_id']
        away_name = row['awayTeam_name']
        
        home_score = row['homeScore']
        away_score = row['awayScore']
        winner = row['winnerCode']
        
        home_stats = get_team_stats(home_id, home_name)
        away_stats = get_team_stats(away_id, away_name)
        
        home_stats['played'] += 1
        away_stats['played'] += 1
        
        home_stats['gf'] += home_score
        home_stats['ga'] += away_score
        away_stats['gf'] += away_score
        away_stats['ga'] += home_score
        
        if winner == 1: # Home win
            home_stats['won'] += 1
            home_stats['points'] += 3
            away_stats['lost'] += 1
        elif winner == 2: # Away win
            away_stats['won'] += 1
            away_stats['points'] += 3
            home_stats['lost'] += 1
        elif winner == 3: # Draw
            home_stats['drawn'] += 1
            home_stats['points'] += 1
            away_stats['drawn'] += 1
            away_stats['points'] += 1

    # Convert to DataFrame
    table_df = pd.DataFrame(teams.values())
    
    # Calculate Goal Difference
    table_df['gd'] = table_df['gf'] - table_df['ga']
    
    # Sort according to UEFA rules (Simplified: Pts, GD, GF)
    table_df = table_df.sort_values(by=['points', 'gd', 'gf'], ascending=[False, False, False]).reset_index(drop=True)
    
    # Add Position
    table_df.insert(0, 'pos', range(1, len(table_df) + 1))
    
    output_file = os.path.join(directory, "uel_2025_26_standings.csv")
    table_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Final standings saved to {output_file}")
    # print(table_df.to_string(index=False))

if __name__ == "__main__":
    main()
