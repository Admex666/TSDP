import argparse
import pandas as pd
import os
from src.scraping.fbref_loader import FBrefDataLoader
from src.features.builder import LineupFeatureBuilder
from src.features.validator import FeatureValidator
from src.betting.backtest import Backtester

def main():
    parser = argparse.ArgumentParser(description="Football Match Prediction Pipeline")
    parser.add_argument("--season", type=str, default="2023-2024", help="Season to process")
    parser.add_argument("--comp", type=str, default="9", help="Competition ID (9=EPL)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches for testing")
    parser.add_argument("--mock", action="store_true", help="Use generated mock data for testing")
    args = parser.parse_args()
    
    # Imports for loop
    from tqdm import tqdm
    import numpy as np
    
    # Handle Mock Data
    data_dir = "data"
    if args.mock:
        print("Running in MOCK mode.")
        data_dir = "data_mock"
        # Generate mocks dynamically
        try:
            import subprocess
            print(f"Generating mock data in {data_dir}...")
            subprocess.check_call(["python", "tests/generate_mock_data.py"])
        except Exception as e:
            print(f"Error calling mock generator: {e}")
            return

    print(f"Initializing pipeline for Season {args.season}...")
    
    # 1. Data Loading
    loader = FBrefDataLoader(data_dir=data_dir)
    print(f"Step 1: Data Ingestion (Source: {data_dir})")
    
    # For mock mode, we know the competition/season might be fake in the URL
    # But generate_mock_data uses the real URL keys "https fbref com..."
    # So the loader's _get_cache_path will find them if we pass the same URLs.
    # The real season/comp args might not match the mock URLs exactly if we don't align them.
    # generate_mock_data uses season "2023-2024" and comp "9", so it should match default args.
    
    schedule = loader.load_match_schedule(args.season, args.comp)
    
    if schedule.empty:
        print("Error: Schedule is empty. Check internet connection or scraper logic.")
        return

    print(f"  - Loaded {len(schedule)} matches.")
    
    # Filter for matches with results for backtesting
    schedule_with_results = schedule.dropna(subset=['HomeGoals', 'AwayGoals']).reset_index(drop=True)
    if args.backtest:
        print(f"  - {len(schedule_with_results)} matches with results available for backtesting.")
        schedule = schedule_with_results # Strict mode for this script
        
    if args.limit:
        print(f"  - Limiting to {args.limit} matches for testing.")
        schedule = schedule.head(args.limit).copy() # Use copy to avoid setting with copy warning
    
    # 2. Feature Engineering
    print("Step 2: Feature Engineering")
    builder = LineupFeatureBuilder(loader)
    
    features_list = []
    # Mocking player positions for MVP since we don't have a full player DB yet
    # In prod, we'd load this from `loader.load_player_season_stats`
    # and map ID -> Pos.
    # For now, we pass empty, so features will default to 0s, 
    # BUT we need to run the loop to generate the matrix structure.
    
    print("  - Building rolling features (This may take time)...")
    
    # Limit for testing if needed
    # process_matches = schedule.head(5) if args.backtest else schedule
    
    # We iterate properly
    for idx, match in tqdm(schedule.iterrows(), total=len(schedule)):
        # We need lineups. 
        # In a real run, we fetch from match['match_report_url']
        # If missing, skip?
        
        match_url = match.get('match_report_url')
        if not match_url:
            # create empty dummy features
            features_list.append({}) 
            continue
            
        # Optimization: Check if features already cached? (Not impl yet)
            
        try:
            lineups = loader.load_match_lineups(match_url)
            
            # Extract player positions from lineups for the builder
            # builder expects a dict {player_id: position_str}
            player_positions = {}
            for side in ['home', 'away']:
                for p in lineups.get(side, []):
                    if 'id' in p and 'position' in p:
                        player_positions[p['id']] = p['position']

            # If lineups empty/failed, we might get 0-vectors
            
            feats = builder.build_features_for_match(
                match, 
                lineups.get('home', []), 
                lineups.get('away', []), 
                player_positions, 
                args.season
            )
            features_list.append(feats)
        except Exception as e:
            print(f"Error processing match {idx}: {e}")
            features_list.append({})
            
    # Create Matrix
    feature_matrix = pd.DataFrame(features_list)
    # Fill any NaNs from empty dicts
    feature_matrix = feature_matrix.fillna(0.0)
    
    # 3. Validation
    print("Step 3: Data Validation")
    validator = FeatureValidator()
    validator.validate(feature_matrix, schedule['Date'])
    
    # 4. Backtesting
    if args.backtest:
        print("Step 4: Running Backtest")
        # Ensure alignment
        if len(feature_matrix) != len(schedule):
            print("Warning: Feature matrix length mismatch. Truncating to min.")
            min_len = min(len(feature_matrix), len(schedule))
            feature_matrix = feature_matrix.iloc[:min_len]
            schedule = schedule.iloc[:min_len]
            
        # Mock Odds for testing if missing
        if 'OddsHome' not in schedule.columns:
            print("  - Injecting synthetic odds for testing.")
            schedule['OddsHome'] = 3.0
            schedule['OddsDraw'] = 10.0 # Force value bet
            schedule['OddsAway'] = 2.4
            
        backtester = Backtester(feature_matrix, schedule)
        backtester.run(start_date="2023-08-05") # Start mid-season
        results = backtester.get_results_df()
        if not results.empty:
            print(results.tail())
            roi = results['Result'].sum() / results['Stake'].sum()
            print(f"Backtest ROI: {roi:.2%}")
        else:
            print("No bets placed.")
        print("  - Backtest complete.")

if __name__ == "__main__":
    main()
