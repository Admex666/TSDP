import sys
from pathlib import Path
import pandas as pd

# Add src path
current_dir = Path(__file__).parent
src_path = current_dir.parent / "src"
sys.path.append(str(src_path))

from scraping.sofascore_loader import SofaScoreLoader
from features.sofascore_builder import SofaScoreFeatureBuilder

def main():
    print("Initializing SofaScore Feature Engineering...")
    
    # 1. Load Loader & Schedule
    loader = SofaScoreLoader(tournament_id=17, season_id=52186)
    df_schedule = loader.load_season_schedule()
    
    # Filter finished matches only for training set, 
    # but the builder handles all (it just won't find history for round 1).
    # Ideally we process ALL so we have features for everything.
    # But targets (scores) might be 0-0 for upcoming.
    
    finished_matches = df_schedule[df_schedule['status'] == 'finished'].copy()
    print(f"Processing {len(finished_matches)} finished matches.")

    # 2. Build Features
    builder = SofaScoreFeatureBuilder(loader, lookback=5)
    df_features = builder.build_features(finished_matches)
    
    # 3. Save
    output_path = Path("data/sofascore/features_sofascore.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    print("\nFeature Engineering Complete!")
    print(f"Saved {len(df_features)} rows to {output_path}")
    print("Columns:", df_features.columns.tolist())
    
    # Show sample
    if not df_features.empty:
        print("\nSample Feature Row (Head):")
        print(df_features.iloc[0])

if __name__ == "__main__":
    main()
