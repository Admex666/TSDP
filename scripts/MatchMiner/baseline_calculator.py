import pandas as pd
import numpy as np

def filter_historical_data(df, target_date):
    """Filters data to exclude future games (strict lookahead bias prevention).
    If no matches exist before the target date, falls back to include matches on/before target_date."""
    hist_df = df[df['Date'] < target_date]
    if hist_df.empty:
        # Fallback to matches on or before target date (first match day of a league)
        hist_df = df[df['Date'] <= target_date]
    return hist_df

def get_team_baselines(df, target_date, league_id):
    """Computes baseline statistics (mean, std, max) for all team metrics in a given league."""
    hist_df = filter_historical_data(df, target_date)
    league_df = hist_df[hist_df['leagueId'] == league_id]
    
    if league_df.empty:
        # Fallback to all leagues if this specific league is empty in history
        league_df = hist_df
        
    baselines = {}
    for col in league_df.select_dtypes(include=[np.number]).columns:
        mean_val = league_df[col].mean()
        std_val = league_df[col].std()
        max_val = league_df[col].max()
        
        # If std is 0 or NaN, default to a very small number to avoid division by zero
        if pd.isna(std_val) or std_val == 0:
            std_val = 1e-5
            
        baselines[col] = {
            'mean': mean_val,
            'std': std_val,
            'max': max_val
        }
    return baselines

def get_player_baselines(df, target_date, league_id):
    """Computes baseline statistics (mean, std, max) for all player metrics in a given league.
    Computes both overall league baselines and position-specific baselines.
    """
    hist_df = filter_historical_data(df, target_date)
    league_df = hist_df[hist_df['newestLeagueId'] == league_id]
    
    if league_df.empty:
        # Fallback if empty
        league_df = hist_df
        
    overall_baselines = {}
    position_baselines = {}  # Nested dict: position -> metric -> stats
    player_historical_stats = {} # Nested dict: playerId -> metric -> stats
    
    # 1. Overall League Player Baselines
    numeric_cols = league_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_val = league_df[col].mean()
        std_val = league_df[col].std()
        max_val = league_df[col].max()
        
        if pd.isna(std_val) or std_val == 0:
            std_val = 1e-5
            
        overall_baselines[col] = {
            'mean': mean_val,
            'std': std_val,
            'max': max_val
        }
        
    # 2. Position-Specific Baselines within League
    # Clean position names first
    league_df = league_df.copy()
    league_df['Position_Clean'] = league_df['Position'].fillna(league_df['pos']).fillna('Unknown')
    
    positions = league_df['Position_Clean'].unique()
    for pos in positions:
        pos_df = league_df[league_df['Position_Clean'] == pos]
        position_baselines[pos] = {}
        for col in numeric_cols:
            mean_val = pos_df[col].mean()
            std_val = pos_df[col].std()
            max_val = pos_df[col].max()
            
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-5
                
            position_baselines[pos][col] = {
                'mean': mean_val,
                'std': std_val,
                'max': max_val
            }
            
    # 3. Individual Player Baselines (to compare player against their own historical average)
    player_ids = league_df['playerId'].unique()
    for p_id in player_ids:
        p_df = league_df[league_df['playerId'] == p_id]
        player_historical_stats[p_id] = {}
        for col in numeric_cols:
            mean_val = p_df[col].mean()
            std_val = p_df[col].std()
            max_val = p_df[col].max()
            
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-5
                
            player_historical_stats[p_id][col] = {
                'mean': mean_val,
                'std': std_val,
                'max': max_val
            }
            
    return overall_baselines, position_baselines, player_historical_stats
