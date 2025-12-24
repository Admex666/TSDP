import sys
import os
import io

# Fix encoding issues for Windows terminal (important for players with special characters in their names)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        # Fallback for environments where reconfigure isn't available
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import sofascore_scraper FIRST to avoid tls_client vs mplsoccer/matplotlib conflict
try:
    from sofascore_scraper import create_lineups_df, fetch_passmap
except ImportError as e:
    print(f"CRITICAL: Local Import Error of 'sofascore_scraper': {e}")
    sys.exit(1)

# Force matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
except ImportError as e:
    print(f"CRITICAL: Import Error: {e}")
    sys.exit(1)

def generate_passing_network(event_id, team_side='home'):
    """
    Generates a passing network visualization for a specific team in a match.
    
    Args:
        event_id (int): SofaScore event ID
        team_side (str): 'home' or 'away'
    """
    print(f"Starting generation for Event {event_id}, Team: {team_side}")
    
    print(f"Fetching lineups for event {event_id}...")
    try:
        lineups_df = create_lineups_df(event_id)
        if lineups_df.empty:
            print("No lineup data found (empty DataFrame). Check Event ID or Internet Connection.")
            return
    except Exception as e:
        print(f"Error fetching lineups: {e}")
        return

    # Filter for the selected team
    team_players = lineups_df[lineups_df['team'] == team_side]
    
    if team_players.empty:
        print(f"No players found for team '{team_side}'. Check team name/side.")
        return

    # We only want starters typically
    starters = team_players[team_players['substitute'] == False]
    
    print(f"Found {len(starters)} starters for {team_side} team.")
    
    all_passes = []
    
    for _, player in starters.iterrows():
        player_name = player.get('name', 'Unknown')
        player_id = player.get('id')
        if not player_id:
            continue
            
        # Use a safe way to print player name to avoid encoding issues
        try:
            print(f"Fetching passmap for {player_name} ({player_id})...")
        except UnicodeEncodeError:
            safe_name = player_name.encode('ascii', 'replace').decode('ascii')
            print(f"Fetching passmap for {safe_name} ({player_id})...")
        
        try:
            pass_df = fetch_passmap(event_id, player_id)
        except Exception as e:
            print(f"Error fetching passmap for {player_name}: {e}")
            continue
        
        if not pass_df.empty:
            passes = pass_df.dropna(subset=['end_x', 'end_y'])
            if 'outcome' in passes.columns:
                passes = passes[passes['outcome'] == 1]
            
            passes = passes.copy() # Avoid SettingWithCopyWarning
            passes['player_name'] = player_name
            passes['player_id'] = player_id
            all_passes.append(passes)
            
    if not all_passes:
        print("No successful pass data found for any player.")
        return

    df_pass = pd.concat(all_passes, ignore_index=True)
    
    # Scale coordinates
    df_pass['x'] = df_pass['player_x'] * 1.2
    df_pass['y'] = df_pass['player_y'] * 0.8
    df_pass['end_x'] = df_pass['end_x'] * 1.2
    df_pass['end_y'] = df_pass['end_y'] * 0.8
    
    # Calculate average positions
    avg_locs = df_pass.groupby('player_name').agg({
        'x': 'mean',
        'y': 'mean',
        'player_name': 'count'
    }).rename(columns={'player_name': 'pass_count'})
    
    print("Generating plot...")
    
    # Setup the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
    fig.set_facecolor('#22312b')
    
    # Plot passes
    pitch.arrows(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y, width=2,
                 headwidth=10, headlength=10, color='#ad993c', ax=ax, label='Passes', alpha=0.5)
    
    # Plot nodes
    pitch.scatter(avg_locs.x, avg_locs.y, s=avg_locs.pass_count*10,
                  color='#ba4f45', edgecolors='#606060', linewidth=2, alpha=1, ax=ax, zorder=3)
    
    for name, row in avg_locs.iterrows():
        name_display = name.split(' ')[-1] if isinstance(name, str) else str(name)
        pitch.annotate(name_display, xy=(row.x, row.y), c='white', va='center',
                       ha='center', size=10, weight='bold', ax=ax, zorder=4)
                       
    plt.title(f'{team_side.title()} Passing Network - Event {event_id}', fontsize=30, color='white')
    
    output_filename = f'pass_map_{event_id}_{team_side}.png'
    try:
        plt.savefig(output_filename, facecolor='#22312b', dpi=300)
        plt.close() # Free memory
        print(f"SUCCESS: Saved plot to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Ensure plot is closed even on error
        try:
            plt.close()
        except:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        e_id = sys.argv[1]
        generate_passing_network(e_id, 'home')
    else:
        print("Usage: python generate_summary.py <event_id>")
