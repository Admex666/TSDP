import pandas as pd
import os
import sys

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from feature_engineer import load_data, calculate_rolling_features, calculate_opponent_strength, OUTPUT_CSV

def main():
    print("1. Loading Data from DB...")
    df = load_data()
    print(f"   Loaded {len(df)} rows.")

    print("2. Calculating Rolling Features (includes Densification)...")
    # This calls reindex_density internally now
    df = calculate_rolling_features(df)
    print(f"   Rows after densification: {len(df)}")

    print("3. Calculating Opponent Strength...")
    df = calculate_opponent_strength(df)

    print(f"4. Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
