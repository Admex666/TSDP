import argparse
import pandas as pd
import os
from src.scraping.sofascore_loader import SofaScoreLoader
from src.features.sofascore_builder import SofaScoreFeatureBuilder
from src.features.validator import FeatureValidator
from src.betting.backtest import Backtester
from src.models.regression import GoalRegressionModel
import glob

def main():
    parser = argparse.ArgumentParser(description="Football Match Prediction Pipeline (SofaScore)")
    parser.add_argument("--seasons", nargs="+", type=int, default=[52186], help="Season IDs to process (e.g., 52186 for 23/24, 41886 for 22/23)")
    parser.add_argument("--tournament", type=int, default=17, help="Tournament ID (17=EPL)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--backfill", action="store_true", help="Run backfill if data missing")
    args = parser.parse_args()
    
    # Imports for loop
    from tqdm import tqdm
    import numpy as np
    
    print(f"Initializing SofaScore pipeline for Tournament {args.tournament}, Seasons {args.seasons}...")
    
    # 1. Data Loading & Compilation
    all_schedules = []
    
    for season_id in args.seasons:
        loader = SofaScoreLoader(tournament_id=args.tournament, season_id=season_id)
        
        # Check if schedule exists, if not and backfill requested, trigger it
        schedule_path = loader.processed_dir / f"schedule_{args.tournament}_{season_id}.csv"
        
        if not schedule_path.exists():
            if args.backfill:
                print(f"Schedule for season {season_id} missing. Backfilling...")
                schedule = loader.load_season_schedule()
                loader.backfill_season(schedule)
            else:
                print(f"Warning: Schedule for season {season_id} not found. Skipping. Use --backfill to fetch.")
                continue
        else:
            schedule = pd.read_csv(schedule_path)
            
        all_schedules.append(schedule)
        
    if not all_schedules:
        print("No data found. Exiting.")
        return
        
    # Combine all seasons
    full_schedule = pd.concat(all_schedules, ignore_index=True)
    # Sort by time
    full_schedule['startTimestamp'] = full_schedule['startTimestamp'].astype(int)
    full_schedule = full_schedule.sort_values('startTimestamp').reset_index(drop=True)
    
    # Add 'Date' column for Backtester compatibility (convert timestamp)
    full_schedule['Date'] = pd.to_datetime(full_schedule['startTimestamp'], unit='s')
    
    print(f"Loaded {len(full_schedule)} matches across {len(args.seasons)} seasons.")
    
    # 2. Feature Engineering
    # We use the loader from the last season iteration, but initialized nicely
    # SofaScoreFeatureBuilder needs *a* loader to fetch match details. 
    # Since loader is initialized with specific season/tourn, does get_match_data work cross-season?
    # Inspecting SofaScoreLoader: it uses self.events_dir which is generic "data/sofascore/processed/events". 
    # It does NOT depend on season_id for *fetching match data* (events are saved by ID). 
    # So any loader instance works.
    
    print("Step 2: Feature Engineering")
    # Use one loader instance for fetching events
    base_loader = SofaScoreLoader(tournament_id=args.tournament, season_id=args.seasons[0])
    builder = SofaScoreFeatureBuilder(base_loader, lookback=5)
    
    feature_df = builder.build_features(full_schedule)
    
    if feature_df.empty:
        print("No features built. Exiting.")
        return
        
    # Fill NaNs
    feature_df = feature_df.fillna(0.0)
    
    # 3. Prepare for Modeling
    print("Step 3: Preparing Data for Models")
    
    # Align Feature Matrix and Schedule
    # builder.build_features returns a DataFrame that HAS match_id.
    # We should merge with full_schedule to ensure we have Odds, etc.
    
    # Rename columns in full_schedule to match Backtester expectations
    # Backtester needs: Date, Home, Away, HomeGoals, AwayGoals, OddsHome, OddsDraw, OddsAway
    
    # Create a clean metadata df
    metadata = full_schedule.copy()
    metadata = metadata.rename(columns={
        'home_team': 'Home',
        'away_team': 'Away',
        'home_score': 'HomeGoals', 
        'away_score': 'AwayGoals'
    })
    
    # We need Odds. 
    # SofaScoreLoader.get_match_data fetches odds, but are they in the schedule?
    # Inspecting SofaScoreLoader.load_season_schedule: NO odds in schedule CSV.
    # Inspecting SofaScoreFeatureBuilder: It builds features but doesn't explicitly return odds columns in the feature DF (it returns target goals).
    
    # We need to extract Odds for the Backtester.
    # Let's do a quick pass to fetch Odds from the 'processed/events/{id}/odds.csv' files and merge them.
    
    print("  - Merging Odds data...")
    odds_data = []
    
    # Optimization: Loading thousands of small CSVs is slow. 
    # In a real heavy pipeline, we'd aggregate this earlier.
    # For now, we iterate.
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Loading Odds"):
        match_id = row['id']
        odds_file = base_loader.events_dir / str(match_id) / "odds.csv"
        
        o_home, o_draw, o_away = 1.0, 1.0, 1.0 # Defaults if missing
        
        if odds_file.exists():
            try:
                df_odds = pd.read_csv(odds_file)
                # content: name, odds, prob...
                # We need "Full time" 1X2. The file usually contains just that based on loader.
                # structure: name (1, X, 2), odds
                
                # flexible matching
                row_1 = df_odds[df_odds['name'] == '1']
                row_x = df_odds[df_odds['name'] == 'X']
                row_2 = df_odds[df_odds['name'] == '2']
                
                if not row_1.empty: o_home = row_1.iloc[0]['odds']
                if not row_x.empty: o_draw = row_x.iloc[0]['odds']
                if not row_2.empty: o_away = row_2.iloc[0]['odds']
            except:
                pass
                
        odds_data.append({
            'id': match_id,
            'OddsHome': o_home,
            'OddsDraw': o_draw,
            'OddsAway': o_away
        })
        
    df_odds_merged = pd.DataFrame(odds_data)
    metadata = metadata.merge(df_odds_merged, on='id', how='left')
    
    # Merge features with metadata to ensure alignment
    # feature_df has 'match_id', metadata has 'id'
    combined = metadata.merge(feature_df, left_on='id', right_on='match_id', how='inner')
    
    # Separate Features and Metadata again, but aligned
    # Features exclude non-feature columns
    exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'target_home_goals', 'target_away_goals', 
                    'id', 'round', 'startTimestamp', 'status', 'home_team_id', 'away_team_id', 'slug', 'Date',
                    'Home', 'Away', 'HomeGoals', 'AwayGoals', 'OddsHome', 'OddsDraw', 'OddsAway', 'index']
    
    feature_cols = [c for c in combined.columns if c not in exclude_cols]
    
    X = combined[feature_cols]
    # Check if we have valid features
    if X.empty:
        print("No numerical features found after merge.")
        return
        
    print(f"  - Final dataset: {len(combined)} matches, {len(feature_cols)} features.")
    
    # Save features for external scripts (e.g., train_sofascore.py)
    output_path = base_loader.data_dir / "features_sofascore.csv"
    print(f"  - Saving features to {output_path}...")
    combined.to_csv(output_path, index=False)
    
    # 4. Backtesting
    if args.backtest:
        print("Step 4: Running Backtest")
        
        # Ensure 'Date' is datetime (it should be)
        backtester = Backtester(X, combined)
        
        # Start mid-season of the first season? 
        # Or start at the beginning of the *second* season if we have multiple?
        # Let's say we start backtesting after 50 matches (approx 5 rounds) to allow initial training.
        
        start_date = combined['Date'].iloc[50] # 50th match time
        print(f"  - Backtest Start Date: {start_date}")
        
        if len(args.seasons) > 1:
            # If multiple seasons, maybe start at the beginning of the last season?
            # Let's default to a reasonable split.
            pass
            
        backtester.run(start_date=str(start_date))
        
        results = backtester.get_results_df()
        if not results.empty:
            print("\nRecent Bets:")
            print(results.tail())
            roi = results['Result'].sum() / results['Stake'].sum() if results['Stake'].sum() > 0 else 0
            print(f"\nBacktest ROI: {roi:.2%}")
            print(f"Total Profit: {results['Result'].sum():.2f}u")
            
            # Save results
            results.to_csv("backtest_results_sofascore.csv", index=False)
            print("Results saved to backtest_results_sofascore.csv")
        else:
            print("No bets placed.")

if __name__ == "__main__":
    main()
