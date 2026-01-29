import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local utils
from data_collector import get_matches_for_round
from feature_engineer import load_data, calculate_rolling_features, calculate_opponent_strength

# Config
SEASON_ID = 76986 # 24/25 Season matching data_collector.py
UNIQUE_TOURNAMENT_ID = 17 # PL
ROUND_TO_PREDICT = 24
MODEL_PATH = os.path.join(current_dir, 'models', 'fantasy_model.json')
PRICES_CSV = os.path.join(current_dir, 'player_prices.csv')
OUTPUT_CSV = os.path.join(current_dir, 'predictions_r24.csv')

def get_next_fixtures(round_num):
    print(f"Fetching fixtures for Round {round_num} (Season {SEASON_ID})...")
    events = get_matches_for_round(UNIQUE_TOURNAMENT_ID, SEASON_ID, round_num)
    
    if not events:
        print("No events found for this round.")
        return pd.DataFrame()
        
    fixtures = []
    for m in events:
        home_team = m['homeTeam']
        away_team = m['awayTeam']
        
        # Add Home Team Fixture info
        fixtures.append({
            'team_id': home_team['id'],
            'team_name': home_team['name'],
            'opponent_team_id': away_team['id'],
            'opponent_name': away_team['name'],
            'is_home': 1,
            'match_id': m['id'],
            'start_timestamp': m.get('startTimestamp')
        })
        
        # Add Away Team Fixture info
        fixtures.append({
            'team_id': away_team['id'],
            'team_name': away_team['name'],
            'opponent_team_id': home_team['id'],
            'opponent_name': home_team['name'],
            'is_home': 0,
            'match_id': m['id'],
            'start_timestamp': m.get('startTimestamp')
        })
        
    return pd.DataFrame(fixtures)

def predict_round():
    print("1. Loading historical data...")
    df_history = load_data()
    
    if df_history.empty:
        print("Error: No historical data loaded. Run data_collector.py first.")
        return

    # EXCLUDE target round from history to prevent duplicates (since we add dummies)
    # This ensures we predict "Next Round" based on previous data only.
    df_history = df_history[df_history['round'] < ROUND_TO_PREDICT] 

    print(f"   Loaded {len(df_history)} rows of history (Pre-Round {ROUND_TO_PREDICT}).")

    print(f"2. Fetching Fixtures for Round {ROUND_TO_PREDICT}...")
    fixtures_df = get_next_fixtures(ROUND_TO_PREDICT)
    
    if fixtures_df.empty:
        print("No fixtures found. Exiting.")
        return

    print(f"   Found {len(fixtures_df)} team-fixtures (matches x 2).")

    print("3. Preparing 'future' rows for feature generation...")
    # New strategy: Append dummy rows for the next match for every active player.
    # Then run rolling calculations on the whole set.
    
    # Identify active players: played in the last 5 rounds or exist in prices
    # We'll take the latest row for every player in history
    latest_player_rows = df_history.sort_values('date').groupby('player_id').tail(1)
    
    team_fixture_map = fixtures_df.set_index('team_id').to_dict('index')
    
    dummies = []
    
    for _, p_row in latest_player_rows.iterrows():
        pid = p_row['player_id']
        tid = p_row['team_id']
        
        # Check if this team plays in next round
        if tid not in team_fixture_map:
            continue
            
        fix = team_fixture_map[tid]
        
        # Mock future date (based on fixture timestamp or +7 days)
        future_date = pd.to_datetime(fix.get('start_timestamp', 0), unit='s')
        if future_date.year < 2000: # Invalid timestamp
            future_date = pd.Timestamp.now() + pd.Timedelta(days=1)

        d = {
            'player_id': pid,
            'player_name': p_row['player_name'],
            'position': p_row['position'],
            'team_id': tid,
            'match_id': fix['match_id'],
            'round': ROUND_TO_PREDICT,
            'date': future_date,
            'opponent_team_id': fix['opponent_team_id'],
            'is_home': fix['is_home'],
            
            # Target columns set to NaN so they don't affect previous rolling avgs (if we were looking back)
            # But vitally, they are NaN because we want to PREDICT them.
            'total_points': np.nan, 
            'minutes': np.nan,
            'goals': np.nan,
            'assists': np.nan,
            'rating': np.nan
        }
        dummies.append(d)
        
    if not dummies:
        print("No active players found matching the upcoming fixtures.")
        return
        
    df_dummies = pd.DataFrame(dummies)
    print(f"   Created {len(df_dummies)} dummy rows for prediction.")
    
    # Concatenate history + future
    full_df = pd.concat([df_history, df_dummies], ignore_index=True)
    
    # Run Feature Engineering
    print("4. Calculating Features (Rolling & Opponent Strength)...")
    full_df = calculate_rolling_features(full_df)
    full_df = calculate_opponent_strength(full_df)
    
    # Filter back to just the future rows
    df_pred = full_df[full_df['round'] == ROUND_TO_PREDICT].copy()
    
    # Prepare for Model: One-Hot Encoding
    print("5. Preparing Model Inputs...")
    # Clean position just in case (e.g. Fwd -> F) - assuming it's already clean standard
    
    # BACKUP Position for output
    df_pred['position_orig'] = df_pred['position']
    
    df_pred = pd.get_dummies(df_pred, columns=['position'], prefix='pos')
    for p in ['pos_G', 'pos_D', 'pos_M', 'pos_F']:
        if p not in df_pred.columns:
            df_pred[p] = 0
            
    # Feature Columns (Must match Training)
    # Feature Columns (Must match Training)
    # START MODEL FEATURES
    start_features = ['starts_last_5', 'last_minutes', 'avg_minutes_last_3', 'round']
    
    # POINTS MODEL FEATURES (Updated list)
    points_features = [
        'round', 'is_home', 
        'last_total_points', 'last_minutes', 
        'last_goals', 'last_assists', 'last_rating',
        'avg_total_points_last_3', 'avg_total_points_last_5', 'avg_total_points_last_38',
        'avg_minutes_last_3', 'avg_minutes_last_5', 'avg_minutes_last_38',
        'avg_goals_last_3', 'avg_goals_last_5', 'avg_goals_last_38',
        'avg_assists_last_3', 'avg_assists_last_5', 'avg_assists_last_38',
        'avg_rating_last_3', 'avg_rating_last_5', 'avg_rating_last_38',
        'ema_points_span_3', 'ema_points_span_5',
        'starts_last_5', 'xFP_weighted', 'form_vs_season',
        'opp_pos_avg_points_allowed'
    ]
    # Add pos cols
    pos_cols = [c for c in df_pred.columns if c.startswith('pos_')]
    points_features.extend(pos_cols)
    
    # Ensure all cols exist
    for c in points_features:
        if c not in df_pred.columns: df_pred[c] = 0
    for c in start_features:
        if c not in df_pred.columns: df_pred[c] = 0
            
    # Load Models
    START_MODEL_PATH = os.path.join(current_dir, 'models', 'start_model.json')
    POINTS_MODEL_PATH = MODEL_PATH # Already defined as models/fantasy_model.json? Check imports.
    
    print(f"6. Loading Models...")
    if not os.path.exists(START_MODEL_PATH) or not os.path.exists(POINTS_MODEL_PATH):
        print("Model files not found! Train them first.")
        return

    start_model = xgb.XGBClassifier()
    start_model.load_model(START_MODEL_PATH)
    
    points_model = xgb.XGBRegressor()
    points_model.load_model(POINTS_MODEL_PATH)
    
    # Predict
    print("   Running prediction (Weighted: P(Start) * Pts)...")
    
    # 1. Probability
    X_start = df_pred[start_features]
    probs = start_model.predict_proba(X_start)[:, 1]
    df_pred['prob_play'] = probs
    
    # 2. Points
    X_points = df_pred[points_features]
    raw_preds = points_model.predict(X_points)
    df_pred['pred_points_raw'] = raw_preds
    
    # 3. Combine
    df_pred['predicted_points'] = df_pred['pred_points_raw'] * df_pred['prob_play']
    
    # Merge Prices
    print(f"7. Merging Prices from {PRICES_CSV}...")
    if os.path.exists(PRICES_CSV):
        prices_df = pd.read_csv(PRICES_CSV)
        # Normalize names for merging (simple lower case strip)
        # Use slug if available in both, but history might not have slug.
        # Fallback to Name.
        
        # It's better to fuzzy match, but let's try direct Name match first
        # Rename 'name' in prices to 'player_name' to match df_pred or vice versa
        prices_df.rename(columns={'name': 'player_name_prices'}, inplace=True)
        
        # Merge
        df_pred = df_pred.merge(prices_df, left_on='player_name', right_on='player_name_prices', how='left')
        df_pred['price'] = df_pred['price'].fillna(0)
        
        # Ensure team name is filled from prices if possible, or fallback
        if 'team' in df_pred.columns:
            df_pred['team'] = df_pred['team'].fillna('Unknown')
    else:
        print("   Prices file not found. Value will be unavailable.")
        df_pred['price'] = 0
        df_pred['team'] = 'Unknown'

    # Determine if it's a Double Gameweek for any player
    # Group by player and Sum points
    print("   Aggregating predictions per player (checking for Double Gameweeks)...")

    # Ensure required columns exist for aggregation
    if 'team' not in df_pred.columns:
        df_pred['team'] = 'Unknown'
        
    final_output = df_pred.groupby(['player_name', 'team_id']).agg({
        'predicted_points': 'sum',
        'pred_points_raw': 'sum',
        'prob_play': 'mean', # Average probability over matches
        'round': 'count', # Count of matches
        'price': 'max', # Price should be constant
        'team': 'first',
        'position_orig': 'first',
        'starts_last_5': 'mean',
        'form_vs_season': 'mean'
    }).reset_index()
    
    final_output.rename(columns={'round': 'matches_count', 'position_orig': 'position'}, inplace=True)
    
    # Recalculate Value on the aggregated score
    final_output['value'] = np.where(final_output['price'] > 0, final_output['predicted_points'] / final_output['price'], 0)
    
    # Sort
    final_output = final_output.sort_values('predicted_points', ascending=False)
    
    print("\nTop 25 Predicted Players (Aggregated by Points):")
    print(final_output[['player_name', 'matches_count', 'prob_play', 'predicted_points', 'price', 'value']].head(25).to_string(index=False))
    
    print("\n" + "="*60)
    print("--- TOP 10 VALUE PICKS BY POSITION ---")
    print("="*60)
    for pos in ['G', 'D', 'M', 'F']:
        pos_df = final_output[final_output['position'] == pos].sort_values('value', ascending=False).head(10)
        print(f"\nPOSITION: {pos}")
        print("-" * 20)
        if not pos_df.empty:
            print(pos_df[['player_name', 'matches_count', 'prob_play', 'predicted_points', 'price', 'value']].to_string(index=False))
        else:
            print("No players found for this position.")

    final_output.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full aggregated predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    predict_round()
