import pandas as pd
import os

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    matches_file = os.path.join(directory, "uel_2025_26_rounds_1-8.csv")
    standings_file = os.path.join(directory, "uel_2025_26_standings.csv")
    
    if not os.path.exists(matches_file) or not os.path.exists(standings_file):
        print("Required files not found.")
        return
        
    df_matches = pd.read_csv(matches_file)
    df_standings = pd.read_csv(standings_file)
    
    # Map team_id to final points
    points_map = df_standings.set_index('team_id')['points'].to_dict()
    
    # Dictionary to store list of ADJUSTED opponent points for each team
    # Map: team_id -> list of (Opponent final points - Opponent points vs team)
    team_opponents_adj = {}
    
    for _, row in df_matches.iterrows():
        h_id = row['homeTeam_id']
        a_id = row['awayTeam_id']
        winner = row['winnerCode']
        
        # Calculate points earned by home and away in THIS match
        h_pts_in_match = 0
        a_pts_in_match = 0
        if winner == 1:
            h_pts_in_match = 3
        elif winner == 2:
            a_pts_in_match = 3
        elif winner == 3:
            h_pts_in_match = 1
            a_pts_in_match = 1
            
        if h_id not in team_opponents_adj: team_opponents_adj[h_id] = []
        if a_id not in team_opponents_adj: team_opponents_adj[a_id] = []
        
        # For Home team: opponent's (Away) points excluding this match
        adj_points_away = points_map.get(a_id, 0) - a_pts_in_match
        team_opponents_adj[h_id].append(adj_points_away)
        
        # For Away team: opponent's (Home) points excluding this match
        adj_points_home = points_map.get(h_id, 0) - h_pts_in_match
        team_opponents_adj[a_id].append(adj_points_home)
        
    # Calculate average adjusted opponent points (Refined SOS)
    sos_results = []
    for team_id, adj_opp_points in team_opponents_adj.items():
        avg_adj_opp_pts = sum(adj_opp_points) / len(adj_opp_points) if adj_opp_points else 0
        sos_results.append({
            'team_id': team_id,
            'avg_opp_points_adj': round(avg_adj_opp_pts, 2)
        })
        
    df_sos = pd.DataFrame(sos_results)
    
    # Merge with standings
    df_final = df_standings.merge(df_sos, on='team_id')
    
    # Sort by Refined SOS (hardest schedule first)
    df_final = df_final.sort_values(by='avg_opp_points_adj', ascending=False)
    
    output_file = os.path.join(directory, "uel_2025_26_sos_refined_analysis.csv")
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Refined SOS Analysis saved to {output_file}")
    
    # Display top 10 hardest and top 10 easiest schedules
    print("\nRefined Hardest Schedules (Excluding direct match points):")
    print(df_final[['pos', 'team_name', 'points', 'avg_opp_points_adj']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
