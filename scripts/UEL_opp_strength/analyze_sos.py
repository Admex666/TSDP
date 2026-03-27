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
    
    # Dictionary to store list of opponent points for each team
    team_opponents = {}
    
    for _, row in df_matches.iterrows():
        h_id = row['homeTeam_id']
        a_id = row['awayTeam_id']
        
        if h_id not in team_opponents: team_opponents[h_id] = []
        if a_id not in team_opponents: team_opponents[a_id] = []
        
        # Add opponent's final points
        team_opponents[h_id].append(points_map.get(a_id, 0))
        team_opponents[a_id].append(points_map.get(h_id, 0))
        
    # Calculate average opponent points (SOS)
    sos_results = []
    for team_id, opp_points in team_opponents.items():
        avg_opp_pts = sum(opp_points) / len(opp_points) if opp_points else 0
        sos_results.append({
            'team_id': team_id,
            'avg_opp_points': round(avg_opp_pts, 2)
        })
        
    df_sos = pd.DataFrame(sos_results)
    
    # Merge with standings
    df_final = df_standings.merge(df_sos, on='team_id')
    
    # Sort by SOS (hardest schedule first)
    df_final = df_final.sort_values(by='avg_opp_points', ascending=False)
    
    output_file = os.path.join(directory, "uel_2025_26_sos_analysis.csv")
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"SOS Analysis saved to {output_file}")

if __name__ == "__main__":
    main()
