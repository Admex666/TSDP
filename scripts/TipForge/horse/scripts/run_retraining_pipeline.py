import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from scripts.merge_odds_results import merge_data
from scripts.prepare_features import engineer_features
from scripts.simulate_walk_forward_variants import main as run_walk_forward_variants

def main():
    print("==========================================================")
    print("      RE-TRAINING & SIMULATION VALIDATION PIPELINE")
    print("==========================================================")
    
    # Step 1: Merge historical results with closing odds
    print("\n[STEP 1/3] Merging results with odds...")
    merge_data()
    
    # Step 2: Engineer features for the full dataset (V4 schema)
    print("\n[STEP 2/3] Engineering features (training_set_v4.csv)...")
    engineer_features()
    
    # Step 3: Run the 2024 validation and 2025 Walk-Forward simulations
    print("\n[STEP 3/3] Running 2024 validation and 2025 Walk-Forward simulations...")
    run_walk_forward_variants()
    
    print("\n==========================================================")
    print("   RE-TRAINING & VALIDATION PIPELINE COMPLETED!")
    print("==========================================================")

if __name__ == "__main__":
    main()
