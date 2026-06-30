import pandas as pd
import numpy as np
from config import TEAM_METRICS, PLAYER_METRICS, PERCENTAGE_METRICS, WEIGHTS
from baseline_calculator import get_team_baselines, get_player_baselines

def analyze_game(team_df, player_df, target_game_id, min_player_minutes=15, min_z_score=1.5, per90=False):
    """Analyzes a specific match and extracts the top team and player insights.
    
    Args:
        team_df: Cleaned DataFrame of team match-level data.
        player_df: Cleaned DataFrame of player match-level data.
        target_game_id: The gameId to analyze.
        min_player_minutes: Minimum minutes a player must play to be considered.
        min_z_score: Minimum Z-score threshold for an insight to be considered.
        per90: If True, scales player volume stats and baselines to per 90 minute equivalents.
        
    Returns:
        dict: containing 'team_insights' and 'player_insights' lists.
    """
    # 1. Locate the target game info
    target_teams = team_df[team_df['gameId'] == target_game_id]
    if target_teams.empty:
        return {"error": f"Game ID {target_game_id} not found in team data."}
        
    # Get date and leagueId from the first team row
    sample_row = target_teams.iloc[0]
    target_date = sample_row['Date']
    league_id = sample_row['leagueId']
    league_name = sample_row.get('leagueName', 'Tournament')
    
    # 2. Scale player data to per90 if requested
    player_df_processed = player_df.copy()
    if per90:
        # Scale only volume metrics (not in PERCENTAGE_METRICS)
        volume_cols = [col for col in PLAYER_METRICS.values() if col not in PERCENTAGE_METRICS and col in player_df_processed.columns]
        
        # We only scale records where Min > 0
        mask = player_df_processed['Min'] > 0
        for col in volume_cols:
            player_df_processed.loc[mask, col] = (player_df_processed.loc[mask, col] * 90.0) / player_df_processed.loc[mask, 'Min']
            
    # 3. Compute baselines using data BEFORE target_date
    team_baselines = get_team_baselines(team_df, target_date, league_id)
    player_league_base, player_pos_base, player_hist_base = get_player_baselines(player_df_processed, target_date, league_id)
    
    # 4. Analyze Team Insights
    team_insights = []
    for _, row in target_teams.iterrows():
        team_name = row['Team']
        opponent_name = row['opponentFullName']
        
        # Determine if it's a knockout match based on date (proxy for World Cup)
        is_knockout = False
        if "world" in str(league_name).lower() and target_date >= pd.to_datetime("2026-06-25"):
            is_knockout = True
            
        for metric_key, col in TEAM_METRICS.items():
            if col not in row or pd.isna(row[col]):
                continue
                
            value = row[col]
            baseline = team_baselines.get(col, {'mean': 0, 'std': 1, 'max': 0})
            
            mean_val = baseline['mean']
            std_val = baseline['std']
            max_val = baseline['max']
            
            # Z-score calculation
            z_score = (value - mean_val) / std_val
            
            # Skip if deviation is not significant enough
            if abs(z_score) < min_z_score:
                continue
                
            is_record = value >= max_val if max_val is not None else False
            
            # Calculate Scores
            rarity_score = min(100.0, abs(z_score) * 20.0)
            record_score = 100.0 if is_record else 0.0
            
            # Context score: knockout bonus, or big away win bonus
            context_score = 0.0
            if is_knockout:
                context_score += 50.0
            if row.get('Away') == True and row.get('Win') == True:
                context_score += 30.0
            context_score = min(100.0, context_score)
            
            # Final interest score
            final_score = (
                rarity_score * WEIGHTS['rarity'] +
                record_score * WEIGHTS['record'] +
                context_score * WEIGHTS['context']
            )
            
            team_insights.append({
                "team": team_name,
                "opponent": opponent_name,
                "metric": metric_key,
                "value": value,
                "mean": mean_val,
                "z_score": z_score,
                "is_record": is_record,
                "score": round(final_score, 1)
            })
            
    # Sort and take top insights for teams
    team_insights = sorted(team_insights, key=lambda x: x['score'], reverse=True)
    
    # 5. Analyze Player Insights
    player_insights = []
    target_players = player_df_processed[player_df_processed['gameId'] == target_game_id]
    
    for _, row in target_players.iterrows():
        # Check minutes restriction
        minutes = row.get('Min', 0)
        if pd.isna(minutes) or minutes < min_player_minutes:
            continue
            
        player_name = row['Player']
        player_id = row['playerId']
        team_name = row['teamName']
        opponent_name = row.get('opponent', 'Opponent')
        pos_raw = row.get('Position', row.get('pos', 'Unknown'))
        position = str(pos_raw) if not pd.isna(pos_raw) else 'Unknown'
        
        # Determine context factors
        is_sub = minutes < 45  # proxy for substitute
        is_knockout = False
        if "world" in str(league_name).lower() and target_date >= pd.to_datetime("2026-06-25"):
            is_knockout = True
            
        for metric_key, col in PLAYER_METRICS.items():
            if col not in row or pd.isna(row[col]):
                continue
                
            value = row[col]
            
            # Don't create insights for 0 counts unless it's a negative insight
            if value == 0 and metric_key not in ["GoalCncd", "TakeOnFail", "FoulCom"]:
                continue
                
            # Get league and position baselines
            league_base = player_league_base.get(col, {'mean': 0, 'std': 1, 'max': 0})
            
            # Position baseline fallback
            pos_base = player_pos_base.get(position, {}).get(col, league_base)
            
            # Individual player baseline (if exists)
            player_base = player_hist_base.get(player_id, {}).get(col, None)
            
            # Z-score calculations
            z_score_league = (value - league_base['mean']) / league_base['std']
            z_score_pos = (value - pos_base['mean']) / pos_base['std']
            
            z_score_player = np.nan
            if player_base is not None and player_base['std'] > 1e-5:
                z_score_player = (value - player_base['mean']) / player_base['std']
                
            # Use position Z-score as primary rarity
            primary_z = z_score_pos if not pd.isna(z_score_pos) else z_score_league
            
            # Skip if deviation is not significant
            if abs(primary_z) < min_z_score:
                continue
                
            is_record = value >= league_base['max'] if league_base['max'] is not None else False
            
            # Rarity Score (max out at 100 for Z >= 5.0)
            rarity_score = min(100.0, abs(primary_z) * 20.0)
            record_score = 100.0 if is_record else 0.0
            
            # Context score
            context_score = 0.0
            if is_sub:
                context_score += 40.0  # Big bonus for doing this in short time!
            if is_knockout:
                context_score += 30.0
            if row.get('Win') == True:
                context_score += 20.0
            context_score = min(100.0, context_score)
            
            final_score = (
                rarity_score * WEIGHTS['rarity'] +
                record_score * WEIGHTS['record'] +
                context_score * WEIGHTS['context']
            )
            
            player_insights.append({
                "player": player_name,
                "position": position,
                "team": team_name,
                "opponent": opponent_name,
                "metric": metric_key,
                "value": value,
                "league_mean": league_base['mean'],
                "pos_mean": pos_base['mean'],
                "player_mean": player_base['mean'] if player_base else np.nan,
                "z_score_pos": z_score_pos,
                "z_score_league": z_score_league,
                "z_score_player": z_score_player,
                "is_record": is_record,
                "score": round(final_score, 1),
                "minutes": minutes,
                "per90": per90
            })
            
    # Sort and take top insights for players
    player_insights = sorted(player_insights, key=lambda x: x['score'], reverse=True)
    
    return {
        "team_insights": team_insights,
        "player_insights": player_insights
    }
