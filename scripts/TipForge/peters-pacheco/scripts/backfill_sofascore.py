import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from scraping.sofascore_loader import SofaScoreLoader

import argparse

def main():
    parser = argparse.ArgumentParser(description="Backfill SofaScore data for multiple seasons.")
    parser.add_argument("--seasons", nargs="+", type=int, help="List of Season IDs to backfill (e.g. 52186 41886)", default=[52186])
    args = parser.parse_args()
    
    print("Initializing SofaScore Backfill...")
    print(f"Seasons to process: {args.seasons}")
    
    for season_id in args.seasons:
        print(f"\n--- Processing Season ID: {season_id} ---")
        # Premier League: 17
        loader = SofaScoreLoader(tournament_id=17, season_id=season_id)
        
        # 1. Load Schedule
        print("Step 1: Loading Schedule...")
        try:
            df_schedule = loader.load_season_schedule()
            print(f"Schedule loaded: {len(df_schedule)} matches.")
            
            # 2. Backfill Matches
            print("Step 2: Backfilling Match Details (Lineups, Stats)...")
            loader.backfill_season(df_schedule)
            
        except Exception as e:
            print(f"Failed to process season {season_id}: {e}")
            
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()
