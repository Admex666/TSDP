import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings

# Suppress pandas concat warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SofaScoreFeatureBuilder:
    def __init__(self, loader, lookback: int = 5):
        """
        :param loader: Instance of SofaScoreLoader
        :param lookback: Number of past matches to consider for rolling averages
        """
        self.loader = loader
        self.lookback = lookback
        
        # Cache for player history to avoid re-reading thousands of CSVs repeatedly
        # Structure: { player_id: List[pd.Series] (rows of stats) }
        # NOTE: This might consume memory. If too large, we rely on disk reading or an intermediate database.
        # For ~500 players * ~38 matches, it's small (20k rows total). Memory is fine.
        self.player_history_cache = {} 
        
        # Pre-load all available data? 
        # Or build it incrementally as we iterate through the schedule?
        # Incremental is safer for "strict past-only" logic verification.

    def build_features(self, df_schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through the schedule chronologically and builds features for each match.
        """
        # Sort by time
        df_schedule = df_schedule.sort_values('startTimestamp').reset_index(drop=True)
        
        features_list = []
        
        # We need to build up history as we go.
        # To make this efficient, we can't fully rebuild history every time.
        # Strategy:
        # 1. Iterate matches.
        # 2. For current match, valid history is everything BEFORE this match's timestamp.
        # 3. Calculate features for Home/Away teams using current history.
        # 4. AFTER calculating feats, add this match's player stats to history for FUTURE matches.
        
        print("Building features from SofaScore data...")
        
        for idx, row in tqdm(df_schedule.iterrows(), total=len(df_schedule), desc="Feature Engineering"):
            match_id = row['id']
            match_time = row['startTimestamp']
            
            # 1. Get Lineups (Target players)
            match_data = self.loader.get_match_data(match_id)
            if 'lineups' not in match_data or match_data['lineups'].empty:
                # Can't build features without knowing who played
                continue
                
            df_lineups = match_data['lineups']
            
            # Filter for starters (substitute = False) usually? 
            # Or weighted? The original paper uses starters.
            starters = df_lineups[df_lineups['substitute'] == False]
            
            home_starters = starters[starters['team'] == 'home']
            away_starters = starters[starters['team'] == 'away']
            
            # 2. Calculate Team Vectors
            home_features = self._aggregate_team_features(home_starters, match_time)
            away_features = self._aggregate_team_features(away_starters, match_time)
            
            # 3. Combine
            match_features = {
                'match_id': match_id,
                'date': match_time,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'target_home_goals': row['home_score'],
                'target_away_goals': row['away_score']
            }
            
            # Add prefix
            for k, v in home_features.items():
                match_features[f'home_{k}'] = v
            for k, v in away_features.items():
                match_features[f'away_{k}'] = v
                
            features_list.append(match_features)
            
            # 4. Update History (feed this match's results into the cache for next iterations)
            # We add ALL players (starters + subs) to history, as even subs earn stats
            self._update_history(df_lineups, match_time)
            
        return pd.DataFrame(features_list)

    def _update_history(self, df_lineups: pd.DataFrame, match_timestamp: int):
        """
        Adds player rows to the history cache.
        """
        # Ensure 'id' column exists for player ID
        if 'id' not in df_lineups.columns:
            return

        # We need to store timestamp to filter later (though incremental loop guarantees past-only,
        # ensuring robustness doesn't hurt).
        df_lineups['match_timestamp'] = match_timestamp
        
        # Relevant stats columns to cache
        stats_cols = [
            'rating', 'totalShots', 'goals', 'expectedGoals', 'keyPass', 
            'totalPass', 'accuratePass', 'totalTackle', 'interceptionWon', 
            'saves', 'goalsPrevented', 'wasFouled', 'fouls', 'totalContest', 
            'wonContest', 'possessionLostCtrl', 'minutesPlayed'
        ]
        
        # Iterate and store
        for _, player_row in df_lineups.iterrows():
            pid = player_row['id']
            stats = {}
            stats['match_timestamp'] = match_timestamp
            
            # extract safe values
            for col in stats_cols:
                # If column exists, take value. If NaN or missing, 0.
                if col in player_row:
                    val = player_row[col]
                    stats[col] = val if pd.notna(val) else 0.0
                else:
                    stats[col] = 0.0
            
            if pid not in self.player_history_cache:
                self.player_history_cache[pid] = []
            self.player_history_cache[pid].append(stats)

    def _aggregate_team_features(self, df_team_players: pd.DataFrame, current_match_time: int) -> Dict[str, float]:
        """
        Aggregates the rolling average stats of the 11 players.
        """
        agg_stats = {
            'avg_rating': [],
            'avg_goals': [], # past goal scoring form
            'avg_xg': [],
            'avg_shots': [],
            'avg_pass_acc': [],
            'avg_tackles': [],
            'avg_interceptions': [],
            'avg_creativity': [] # key passes
        }
        
        # We can also stratify by position (GK headers separate?)
        # For simplicity V1: Team Average.
        
        total_players_found = 0
        
        for _, p in df_team_players.iterrows():
            pid = p['id']
            if pid not in self.player_history_cache:
                continue
                
            history = self.player_history_cache[pid]
            # Ensure strict past
            past_games = [g for g in history if g['match_timestamp'] < current_match_time]
            
            if not past_games:
                continue
                
            # Take last N
            recent_games = sorted(past_games, key=lambda x: x['match_timestamp'])[-self.lookback:]
            
            # Calculate averages for this player
            df_hist = pd.DataFrame(recent_games)
            
            # Helper to safe mean
            def safe_mean(col):
                return df_hist[col].mean() if col in df_hist.columns else 0.0

            agg_stats['avg_rating'].append(safe_mean('rating'))
            agg_stats['avg_goals'].append(safe_mean('goals'))
            agg_stats['avg_xg'].append(safe_mean('expectedGoals'))
            agg_stats['avg_shots'].append(safe_mean('totalShots'))
            
            # Pass Acc handling
            if 'totalPass' in df_hist.columns and 'accuratePass' in df_hist.columns:
                # Sum then divide to avoid div/0 in single games
                tot = df_hist['totalPass'].sum()
                acc = df_hist['accuratePass'].sum()
                agg_stats['avg_pass_acc'].append(acc / tot if tot > 0 else 0)
            else:
                agg_stats['avg_pass_acc'].append(0)
                
            agg_stats['avg_tackles'].append(safe_mean('totalTackle'))
            agg_stats['avg_interceptions'].append(safe_mean('interceptionWon'))
            agg_stats['avg_creativity'].append(safe_mean('keyPass'))
            
            total_players_found += 1
            
        # Now aggregate to Team Level (Mean of the 11 players)
        team_features = {}
        for k, v_list in agg_stats.items():
            team_features[k] = np.mean(v_list) if v_list else 0.0
            
        # Add completeness feature (how many players had history?)
        # Could be useful for confidence
        team_features['history_completeness'] = total_players_found / len(df_team_players) if len(df_team_players) > 0 else 0
        
        return team_features
