import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from scraping.sofascore import (
    get_events_for_round,
    create_lineups_df,
    create_average_positions_df,
    create_statistics_df,
    create_shotmap_df,
    create_graph_df,
    create_odds_df,
    fetch_passmap
)

def test_scraper():
    print("Step 1: Fetching Round 1 Events for PL 23/24...")
    # Premier League: 17, Season 23/24: 52186
    events = get_events_for_round(17, 52186, 1)
    
    if not events:
        print("Failed to fetch events.")
        return

    print(f"Found {len(events)} events.")
    
    # Pick the first finished event
    target_event = None
    for event in events:
        if event['status']['type'] == 'finished':
            target_event = event
            break
            
    if not target_event:
        print("No finished events found in Round 1 (weird).")
        return
        
    event_id = target_event['id']
    match_slug = target_event['slug']
    print(f"\nTarget Match: {match_slug} (ID: {event_id})")
    
    # Test Lineups
    print("\n--- Testing Lineups ---")
    df_lineups = create_lineups_df(event_id)
    print(f"Lineups Shape: {df_lineups.shape}")
    if not df_lineups.empty:
        print(df_lineups[['name', 'team', 'position', 'rating']].head())

    # Test Avg Positions
    print("\n--- Testing Avg Positions ---")
    df_pos = create_average_positions_df(event_id)
    print(f"Positions Shape: {df_pos.shape}")
    
    # Test Statistics
    print("\n--- Testing Statistics ---")
    df_stats = create_statistics_df(event_id)
    print(f"Stats Shape: {df_stats.shape}")
    if not df_stats.empty:
        print(df_stats[['group', 'statistic', 'home_display', 'away_display']].head())

    # Test Shotmap
    print("\n--- Testing Shotmap ---")
    df_shots = create_shotmap_df(event_id)
    print(f"Shotmap Shape: {df_shots.shape}")
    if not df_shots.empty:
        print("Shotmap Columns:", df_shots.columns.tolist())
        # Try to find xg column
        xg_col = next((c for c in df_shots.columns if 'xg' in c.lower()), None)
        cols_to_show = ['name', 'shotType']
        if xg_col: cols_to_show.append(xg_col)
        print(df_shots[cols_to_show].head())

    # Test Graph (Momentum)
    print("\n--- Testing Motivation Graph ---")
    df_graph = create_graph_df(event_id)
    print(f"Graph Points: {df_graph.shape}")

    # Test Odds
    print("\n--- Testing Odds ---")
    df_odds = create_odds_df(event_id)
    print(f"Odds Rows: {df_odds.shape}")
    if not df_odds.empty:
        print(df_odds)
        
    # Test Passmap (Sample player)
    if not df_lineups.empty:
        # Pick a player ID
        pid = df_lineups.iloc[0]['id']
        pname = df_lineups.iloc[0]['name']
        print(f"\n--- Testing Passmap for {pname} ({pid}) ---")
        df_pass = fetch_passmap(event_id, pid)
        print(f"Passmap Shape: {df_pass.shape}")


if __name__ == "__main__":
    test_scraper()
