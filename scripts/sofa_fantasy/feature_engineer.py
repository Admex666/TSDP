import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'sofascore_fantasy.db')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'training_data.csv')

# Defaults (can be overridden)
SEASON_ID_DEFAULT = 76986 

def load_data(season_id=SEASON_ID_DEFAULT):
    """Load necessary data for a specific season from DB."""
    conn = sqlite3.connect(DB_PATH)
    
    query = f"""
    SELECT 
        pms.id as stat_id,
        pms.player_id,
        p.name as player_name,
        p.position,
        pms.match_id,
        m.round,
        m.date,
        m.home_team_id,
        m.away_team_id,
        m.season_id,
        pms.team_id,
        CASE WHEN pms.team_id = m.home_team_id THEN m.away_team_id ELSE m.home_team_id END as opponent_team_id,
        CASE WHEN pms.team_id = m.home_team_id THEN 1 ELSE 0 END as is_home,
        pms.minutes,
        pms.total_points,
        pms.goals,
        pms.assists,
        pms.rating
    FROM player_match_stats pms
    JOIN matches m ON pms.match_id = m.id
    JOIN players p ON pms.player_id = p.id
    WHERE m.season_id = {season_id}
    ORDER BY pms.player_id, m.date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    return df

def reindex_density(df):
    """
    Re-index the dataframe so that every player has a row for every round 
    between their first and last appearance (or min/max of dataset).
    This ensures that rolling averages (like starts_last_5) account for MISSED games as 0s.
    """
    # Create a grid of all Rounds per Season? Or just simplified min-max round per player.
    # Simpler: For each player, fill rounds between min(round) and max(round).
    # actually, global min-max is better to capture players who missed start or end.
    
    # Let's do per-player fill from Round 1 to Current Round (or max in data).
    # But wait, we have multiple seasons maybe?
    # Assuming single season for now (Season 61627). 
    # If multiple seasons, we should group by season too.
    # The SQL query doesn't show season_id clearly, but we know we are working on one season usually.
    # Let's assume 'round' is unique enough or we group by player.
    
    # Get all unique rounds present in data
    # all_rounds = sorted(df['round'].unique())
    # Actually, simpler: 
    # Group by player, set index to round, reindex to range(min_round, max_round).
    
    print("Densifying Data (Filling missing rounds with 0)...")
    
    df_dense_list = []
    
    # We need to know the global round range to be fair?
    # Or just individual range? 
    # User said: "last 5 rounds". If I missed round 20, but played 19 and 21, 
    # rolling(3) at 21 should see (21, 0, 19).
    
    min_round = df['round'].min()
    max_round = df['round'].max()
    all_rounds = np.arange(min_round, max_round + 1)
    
    # Pre-calculate round to date mapping for filling Nat dates
    # We take the median date for each round to avoid outliers
    round_dates = df.groupby('round')['date'].apply(lambda x: pd.to_datetime(x).dropna().median()).to_dict()
    
    # Iterate players
    print(f"   --- Densifying data for rounds {min_round} to {max_round} ---")
    
    for pid, group in df.groupby('player_id'):
        # Get static info
        static_info = {
            'player_id': pid,
            'player_name': group['player_name'].iloc[0],
            'position': group['position'].iloc[0],
            'team_id': group['team_id'].iloc[0]
        }
        
        # Deduplicate round (might happen if dummy matches existing)
        group = group.drop_duplicates(subset=['round'], keep='last')
        
        # Set round as index
        g_idx = group.set_index('round')
        
        # Reindex
        g_dense = g_idx.reindex(all_rounds)
        
        # Fill static
        for col, val in static_info.items():
            g_dense[col] = g_dense[col].fillna(val)
            
        # Fill Date using round mapping if missing
        g_dense['date'] = g_dense.apply(
            lambda row: round_dates.get(row.name, row['date']) if pd.isna(row['date']) else row['date'], 
            axis=1
        )
        
        # Fill metrics with 0
        fill_zeros = ['minutes', 'total_points', 'goals', 'assists', 'rating']
        for col in fill_zeros:
            g_dense[col] = g_dense[col].fillna(0)
            
        # Fill date? We need date for sorting.
        # We can interpolate or just not use date for sorting anymore, use Round.
        # But rolling might depend on date sort? 
        # Actually `calculate_rolling_features` sorts by [player_id, date].
        # If date is NaT, problem.
        # Let's fill Round column (index to col)
        g_dense['round'] = g_dense.index
        
        if "Nick Pope" in static_info['player_name']:
            print(f"Debug Pope: History Rounds: {group['round'].tolist()}")
            print(f"Debug Pope: Densified Tail:\n{g_dense.tail(5)}")
        
        df_dense_list.append(g_dense)
        
    df_dense = pd.concat(df_dense_list, ignore_index=True)
    
    # Restore 'date' ? 
    # We can assign a dummy date for the filled rows if needed, or sort by round.
    # Let's assume sorting by Round is sufficient if Date is missing.
    # But for opponent strength we need date order.
    # Let's just Sort by Round.
    
    return df_dense

def calculate_rolling_features(df):
    """Calculate rolling averages for each player."""
    print("Calculating Rolling Features...")
    
    # 1. Densify first (Fill gaps)
    # This ensures that rolling averages (like starts_last_5) account for MISSED games as 0s.
    df = reindex_density(df)
    
    # Sort by player and ROUND
    df = df.sort_values(['player_id', 'round'])
    
    # Metrics to aggregate
    metrics = ['total_points', 'minutes', 'goals', 'assists', 'rating']
    windows = [3, 5, 38] 
    
    df_grouped = df.groupby('player_id')
    
    for m in metrics:
        # Lag 1 (Previous match stats)
        # We need to shift everything by 1 because we can't use current match stats to predict current match points
        
        # 1. Simple Lag (Last Game)
        df[f'last_{m}'] = df_grouped[m].shift(1)
        
        # 2. Rolling Avgs
        for w in windows:
            # rolling(closed='left') is tricky in pandas, usually we do rolling().mean().shift(1)
            # This ensures we don't include current value
            col_name = f'avg_{m}_last_{w}'
            df[col_name] = df_grouped[m].transform(lambda x: x.rolling(window=w, min_periods=1).mean().shift(1))
            
        # 3. EMA (Exponential Moving Average) - Weighted Form
        if m == 'total_points':
             for span in [3, 5]:
                 col_name = f'ema_points_span_{span}'
                 df[col_name] = df_grouped[m].transform(lambda x: x.ewm(span=span, adjust=False).mean().shift(1))

    # --- Availability Features ---
    # Calc ratio of games started (minutes > 0) in last 5
    # We use 'minutes' column. If > 0, they played. 
    # Actually, we want to know if they started. We don't have 'is_starter' in the aggregated DF easily unless we parse it.
    # But 'minutes > 0' is a good proxy for "Played".
    # Let's create a 'played' boolean
    df['played'] = (df['minutes'] > 0).astype(int)
    df['starts_last_5'] = df_grouped['played'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    
    # Interaction: Expected Performance = EMA * Availability Factor
    # If they play 100% of time -> 1.0 * EMA. If 50% -> 0.5 * EMA.
    # We use avg_minutes_last_5 directly as a scalar for volume?
    # Or just start_ratio?
    # Let's use avg_points (EMA) * start_ratio as "Expected Points Adjusted for Reliability"
    df['xFP_weighted'] = df['ema_points_span_5'] * df['starts_last_5']
    
    # Form: Avg points last 3 - Avg points last 38 (short term vs long term)
    df['form_vs_season'] = df['avg_total_points_last_3'] - df['avg_total_points_last_38']
    
    return df

def calculate_opponent_strength(df):
    """Calculate how many points an opponent concedes to a specific position."""
    print("Calculating Opponent Difficulty...")
    
    # We want to know: For this Opponent (opponent_team_id), against this Position (position),
    # what is the average total_points yielded?
    
    # We must be careful not to create leakage. 
    # Opponent strength should be calculated based on PRIOR matches.
    # A simple expanding mean per (opponent, position) could work.
    
    # Create an auxiliary DF for calculation
    # We will compute accumulating stats
    
    # Sort by date
    df = df.sort_values('date')
    
    # We can use expanding mean on the group (opponent, position)
    # But we need to shift passed stats.
    
    # Group by Opponent and Position
    # Target: total_points allowed
    
    df['opp_pos_avg_points_allowed'] = df.groupby(['opponent_team_id', 'position'])['total_points'] \
        .transform(lambda x: x.expanding().mean().shift(1))
        
    # Global average for imputing NaN (first match against a team)
    global_avgs = df.groupby('position')['total_points'].mean()
    
    # Fill NaNs with global position average
    for pos in df['position'].unique():
        mask = (df['position'] == pos) & (df['opp_pos_avg_points_allowed'].isna())
        if pos in global_avgs:
            df.loc[mask, 'opp_pos_avg_points_allowed'] = global_avgs[pos]
            
    return df

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    # 1. Rolling Features (Player Form)
    df = calculate_rolling_features(df)
    
    # 2. Opponent Strength
    df = calculate_opponent_strength(df)
    
    # 3. Clean up
    # Remove rows where we don't have enough history? 
    # Actually XGBoost handles NaNs, but for "Lag 1" we need at least 2nd game.
    # The first game of the season has no history. We can keep it but features will be NaN/Global Avg.
    # Or we can drop Round 1.
    
    # Let's keep distinct columns for export
    feature_cols = [
        'round', 'is_home', 'position', 'opponent_team_id', 'team_id',
        'last_total_points', 'last_minutes', 
        'avg_total_points_last_3', 'avg_total_points_last_5', 'avg_total_points_last_38',
        'ema_points_span_3', 'ema_points_span_5',
        'starts_last_5', 'xFP_weighted',
        'avg_minutes_last_3', 
        'avg_goals_last_5', 'avg_assists_last_5',
        'avg_rating_last_5',
        'form_vs_season',
        'opp_pos_avg_points_allowed'
    ]
    
    target_col = 'total_points'
    meta_cols = ['player_id', 'player_name', 'match_id', 'date']
    
    final_df = df[meta_cols + feature_cols + [target_col]].copy()
    
    # Fill remaining NaNs (e.g. Round 1 has no previous stats) with -1 or 0?
    # For tree models, simple fill or keeping NaNs is option. 
    # Let's fill rolling stats with 0 for Round 1, or maybe the season average?
    # Simple approach: Fill N/A in rolling features with 0 (as if they did nothing before).
    final_df = final_df.fillna(0) 
    
    print(f"Saving {len(final_df)} rows to {OUTPUT_CSV}")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
