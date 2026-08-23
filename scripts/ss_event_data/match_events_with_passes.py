import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "modules")))
import SofaScore_module as ssm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    match_id = 16483643
    live_csv = "live_events_16483643.csv"
    passes_csv = f"player_passes_{match_id}.csv"
    
    # 1. Load cached or fetch passes
    if os.path.exists(passes_csv):
        print(f"Reading player actions from {passes_csv}...")
        df_passes = pd.read_csv(passes_csv)
    else:
        print(f"Fetching all player actions from SofaScore API for match {match_id}...")
        df_passes = ssm.create_all_passes_df(match_id)
        df_passes.to_csv(passes_csv, index=False)
        
    print(f"Loaded {len(df_passes)} player actions.")
    
    # 2. Match live events with player passes
    print(f"Matching live stream events from {live_csv} with player passes...")
    df_matched = ssm.match_live_events_with_player_passes(
        live_events_df_or_path=live_csv,
        player_passes_df=df_passes
    )
    
    print(f"\nSuccessfully matched {len(df_matched)} events!")
    
    # Display overview
    cols_to_show = [
        'time_captured', 'match_minute', 'match_seconds', 'name', 'situation', 
        'team', 'player_name', 'action_type', 'outcome', 'x', 'y', 'match_coord_dist'
    ]
    avail_cols = [c for c in cols_to_show if c in df_matched.columns]
    print("\n--- Enriched Live Event Stream (Sample) ---")
    print(df_matched[avail_cols].to_string())
    
    output_matched_csv = f"matched_events_{match_id}.csv"
    df_matched.to_csv(output_matched_csv, index=False)
    print(f"\nSaved enriched events to {output_matched_csv}")
