import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Set efficient plotting style
sns.set_theme(style="whitegrid")

def analyze_and_plot():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    players_csv = os.path.join(current_dir, "top_players.csv")
    teams_csv = os.path.join(current_dir, "team_goals.csv")
    
    # 1. Load Data
    if not os.path.exists(players_csv) or not os.path.exists(teams_csv):
        print("Error: Missing CSV files. Please run fetch_top_players.py and fetch_top_teams.py first.")
        return

    print("Loading data...")
    df_players = pd.read_csv(players_csv)
    df_teams = pd.read_csv(teams_csv)
    
    # Clean player data (NaN -> 0 for goals/assists)
    df_players['goals'] = df_players['goals'].fillna(0)
    df_players['assists'] = df_players['assists'].fillna(0)
    
    # Calculate player total contribution
    df_players['total_contribution'] = df_players['goals'] + df_players['assists']
    
    # 2. Merge Data
    # Merge player Stats with Team goals
    # Note: 'total_goals' comes from the team API, which is the TRUE total
    merged_df = df_players.merge(df_teams, on='team_name', how='left')
    
    # 3. Calculate True Percentage
    merged_df['contribution_percent'] = (merged_df['total_contribution'] / merged_df['total_goals']) * 100
    
    # 4. Filter & Sort
    # We want the top contributor for each team
    # Sort by team, then contribution desc
    df_sorted = merged_df.sort_values(['team_name', 'total_contribution'], ascending=[True, False])
    
    # Get top player per team
    top_contributors = df_sorted.drop_duplicates(subset=['team_name'], keep='first')
    
    # Sort for plotting (highest percentage first)
    plot_data = top_contributors.sort_values('contribution_percent', ascending=False)
    
    # 5. Save Analysis to CSV
    output_csv = os.path.join(current_dir, "final_contribution_analysis.csv")
    plot_data.to_csv(output_csv, index=False)
    print(f"Analysis saved to: {output_csv}")
    
    # 6. Create Visualization
    print("Creating visualization...")
    plt.figure(figsize=(14, 12))
    
    # Create combined label: "PlayerName (TeamName)"
    plot_data['label'] = plot_data['player_name'] + " (" + plot_data['team_name'] + ")"
    
    # Bar plot
    ax = sns.barplot(
        data=plot_data,
        x='contribution_percent',
        y='label',
        hue='team_name', 
        palette='viridis',
        dodge=False
    )
    
    # Remove legend as it's too big with 36 teams
    if ax.get_legend():
        ax.get_legend().remove()
    
    # Add value labels to bars
    for i, v in enumerate(plot_data['contribution_percent']):
        stats_text = f"{v:.1f}% ({int(plot_data.iloc[i]['total_contribution'])}/{int(plot_data.iloc[i]['total_goals'])})"
        ax.text(v + 0.5, i, stats_text, va='center', fontsize=10, fontweight='bold')
    
    plt.title('Top Goal Contributors by Team (Goals + Assists) / Total Team Goals', fontsize=16, pad=20)
    plt.xlabel('Contribution Percentage (%)', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, max(plot_data['contribution_percent']) + 15) # Add space for labels
    
    plt.tight_layout()
    
    # Save Plot
    plot_path = os.path.join(current_dir, "contribution_chart.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Chart saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    try:
        analyze_and_plot()
    except Exception as e:
        print(f"An error occurred: {e}")
