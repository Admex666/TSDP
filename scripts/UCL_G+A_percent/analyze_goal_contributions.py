import pandas as pd
import os
import sys

# Set encoding to utf-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

def analyze_contributions():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "top_players.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run fetch_top_players.py first.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Data Cleaning
    # Ensure numerical columns are handled correctly (NaN -> 0)
    cols_to_fix = ['goals', 'assists']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
            
    # Calculate individual total contribution (G + A)
    df['total_contribution'] = df['goals'] + df['assists']
    
    # 2. Team Level Aggregation
    # Note: This sums up goals ONLY from the players present in the top_players.csv.
    # If the API didn't return all players who scored, this number might be lower than reality.
    team_stats = df.groupby('team_name')['goals'].sum().reset_index()
    team_stats.rename(columns={'goals': 'team_dataset_goals'}, inplace=True)
    
    # Filter out teams with 0 goals in the dataset to avoid division by zero
    team_stats = team_stats[team_stats['team_dataset_goals'] > 0]
    
    # Merge team stats back to main dataframe
    df = df.merge(team_stats, on='team_name')
    
    # 3. Calculate Percentage Contribution
    df['contribution_percent'] = (df['total_contribution'] / df['team_dataset_goals']) * 100
    
    # 4. Find Top Contributor per Team
    # Sort by team name and then by total contribution (descending)
    df_sorted = df.sort_values(['team_name', 'total_contribution'], ascending=[True, False])
    
    # Get the top player for each team
    top_contributors = df_sorted.drop_duplicates(subset=['team_name'], keep='first')
    
    # Sort results by contribution percentage for better readability
    final_result = top_contributors[[
        'team_name', 
        'team_dataset_goals', 
        'player_name', 
        'goals', 
        'assists', 
        'total_contribution', 
        'contribution_percent'
    ]].sort_values('contribution_percent', ascending=False)
    
    # Formatting for display
    final_result['contribution_percent'] = final_result['contribution_percent'].map('{:.1f}%'.format)
    
    # 5. Output
    print("\n--- Legfőbb Gól + Gólpassz Hozzájárulók Csapatonként ---")
    print("(A % a datasetben szereplő játékosok összes góljához viszonyítva értendő)")
    print("-" * 100)
    print(final_result.to_string(index=False))
    
    # Save analysis to CSV
    output_path = os.path.join(current_dir, "goal_contributions_analysis.csv")
    final_result.to_csv(output_path, index=False)
    print(f"\nElemzés mentve ide: {output_path}")

if __name__ == "__main__":
    analyze_contributions()
