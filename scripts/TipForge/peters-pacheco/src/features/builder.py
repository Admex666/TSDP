import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from ..scraping.fbref_loader import FBrefDataLoader

class LineupFeatureBuilder:
    """
    Constructs lineup-based features using historical player stats.
    Enforces strict past-only data usage to prevent leakage.
    """
    
    def __init__(self, loader: FBrefDataLoader):
        self.loader = loader
        
    def _get_rolling_stats(self, player_id: str, match_date: pd.Timestamp, season: str, metrics: List[str], window: int = 10, log_type: str = "summary") -> Dict[str, float]:
        """
        Compute rolling average stats for a player strictly before match_date.
        """
        logs = self.loader.load_player_match_logs(player_id, season, log_type)
        
        if logs.empty:
            return {m: 0.0 for m in metrics}
            
        # Filter strict past
        # Ensure dates are comparable
        if 'Date' not in logs.columns:
             return {m: 0.0 for m in metrics}
             
        past_logs = logs[logs['Date'] < match_date].sort_values('Date')
        
        if past_logs.empty:
            return {m: 0.0 for m in metrics}
            
        # Take last N matches
        recent_logs = past_logs.tail(window)
        
        # Calculate mean for each metric
        stats = {}
        for metric in metrics:
            # Helper to find column case-insensitive
            col_match = next((c for c in recent_logs.columns if c.lower() == metric.lower() or c.lower().endswith(f"_{metric.lower()}")), None)
            
            if col_match:
                # Force numeric
                numeric_series = pd.to_numeric(recent_logs[col_match], errors='coerce').fillna(0)
                stats[metric] = numeric_series.mean()
            else:
                stats[metric] = 0.0
                
        return stats

    def build_features_for_match(self, match: Dict, home_lineup: List[Dict], away_lineup: List[Dict], player_positions: Dict[str, str], season: str) -> Dict:
        """
        Build aggregated features for both teams based on their lineups.
        """
        match_date = pd.to_datetime(match['Date'])
        features = {}
        
        # Metric Definitions (mapped to likely FBref Match Log columns)
        
        # GK Metrics (need 'keepers' log)
        # Added 'errors': 'Err' which might come from summary/keepers. Check log availability.
        gk_metrics = {
            "clean_sheets": "CS", "goals_against": "GA", "psxg": "PSxG", 
            "psxg_diff": "PSxG+/-", "save_pct": "Save%", "shots_on_target_against": "SoTA", 
            "crosses_stopped_pct": "Stp%", "sweeper_actions": "#OPA", 
            "avg_pass_length": "AvgLen", "pass_completion_pct": "Cmp%", "launch_pct": "Launch%",
            "errors": "Err"
        }
        
        # Outfield Metrics (Summary/Defense/Passing/Possession merged ideally)
        outfield_mapping = {
            # DEF
            "tackles": "Tkl", "interceptions": "Int", "blocks": "Blocks", "clearances": "Clr",
            "aerials_won": "Won", "pass_cmp_pct": "Cmp%", "progressive_passes": "PrgP",
            "progressive_carries": "PrgC", "passes_into_final_third": "1/3", 
            "shots": "Sh", "xg": "xG", "fouls": "Fls", "errors": "Err",
            
            # MID/ATT additional
            "goals": "Gls", "assists": "Ast", "xa": "xAG", "key_passes": "KP",
            "passes_into_box": "PPA", "goal_creating_actions": "GCA", "shot_creating_actions": "SCA", 
            "touches": "Touches", "touches_in_box": "Att Pen", # Att Pen touches
            "fouls_drawn": "Fld", "dribbles_completed": "Succ"
        }
        
        def process_team(lineup, side):
            # Segment players
            groups = {'GK': [], 'DEF': [], 'MID': [], 'ATT': []}
            
            for player in lineup:
                pid = player['id']
                # Simplistic position mapping if not provided
                pos = player_positions.get(pid, 'MF') # Default to MID if unknown
                
                if 'GK' in pos: target = 'GK'
                elif 'DF' in pos: target = 'DEF'
                elif 'FW' in pos: target = 'ATT'
                else: target = 'MID'
                
                groups[target].append(pid)
                
            # Process GK
            if groups['GK']:
                gk_id = groups['GK'][0] # Assume 1 GK
                stats = self._get_rolling_stats(gk_id, match_date, season, list(gk_metrics.values()), log_type="keepers")
                for name, col in gk_metrics.items():
                    features[f"{side}_gk_{name}"] = stats.get(col, 0.0)
            else:
                for name in gk_metrics.keys():
                    features[f"{side}_gk_{name}"] = 0.0

            # Process Outfield Groups
            for grp_name, metric_list in [
                ('def', ["tackles", "interceptions", "blocks", "clearances", "aerials_won", "pass_cmp_pct", "progressive_passes", "progressive_carries", "passes_into_final_third", "shots", "xg", "fouls", "errors"]),
                ('mid', ["goals", "assists", "xg", "xa", "shots", "key_passes", "progressive_passes", "progressive_carries", "pass_cmp_pct", "passes_into_box", "goal_creating_actions", "shot_creating_actions", "touches", "fouls_drawn"]),
                ('att', ["goals", "xg", "shots", "shots_on_target", "assists", "xa", "key_passes", "goal_creating_actions", "shot_creating_actions", "dribbles_completed", "touches_in_box", "progressive_carries", "fouls_drawn"])
            ]:
                pids = groups[grp_name.upper()]
                if not pids:
                    # Fill 0s
                    for m in metric_list:
                        features[f"{side}_{grp_name}_{m}"] = 0.0
                    continue
                    
                # Get stats for all players in group
                group_stats = []
                for pid in pids:
                    cols_needed = [outfield_mapping.get(m, m) for m in metric_list]
                    p_stats = self._get_rolling_stats(pid, match_date, season, cols_needed, log_type="summary")
                    group_stats.append(p_stats)
                    
                # Compute Mean
                if group_stats:
                    df_grp = pd.DataFrame(group_stats)
                    means = df_grp.mean(numeric_only=True)
                    
                    for m_name, col_name in zip(metric_list, cols_needed):
                        features[f"{side}_{grp_name}_{m_name}"] = means.get(col_name, 0.0)
                else:
                    for m in metric_list:
                        features[f"{side}_{grp_name}_{m}"] = 0.0

        process_team(home_lineup, 'home')
        process_team(away_lineup, 'away')
        
        return features
