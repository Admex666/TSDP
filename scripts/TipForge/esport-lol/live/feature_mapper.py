import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class RiotFeatureMapper:
    """
    Maps Riot Esports API JSON data to the 29 features expected by the ML model.
    """
    
    def __init__(self):
        self.feature_cols = [
            'kills_diff', 'towers_diff', 'drakes_diff', 'barons_BLUE', 'barons_RED',
            'gold_diff', 'gold_diff_pct', 'cs_diff', 'gold_per_min_blue', 'gold_per_min_red',
            'cs_per_min_blue', 'cs_per_min_red', 'kill_momentum_3min', 'gold_momentum_5min', 
            'drake_control_score', 'tower_sequence_score', 'baron_timing_advantage', 
            'early_advantage', 'mid_advantage', 'late_advantage', 'has_soul_point', 
            'gold_lead_critical', 'tower_lead_critical', 'phase_early', 'phase_mid',
            'phase_late', 'gold_diff_x_minute', 'kills_diff_x_minute', 'momentum_composite'
        ]

    def map_riot_to_features(self, window_frame: Dict, details_frame: Optional[Dict] = None) -> Dict:
        """
        Calculates features from a Riot window frame and optional details frame.
        """
        blue_team = window_frame.get('blueTeam', {})
        red_team = window_frame.get('redTeam', {})
        
        # Game time from timestamp or assumed (Riot window frames often come in sequence)
        # For now, we'll try to estimate minute from frame index or metadata if available
        # In a real live loop, we track the elapsed time.
        # Let's assume we pass the minute externally or extract it.
        # Note: Riot frames are usually 1 minute or 10s intervals.
        
        # For the sake of calculations in this mapper, we need the current minute.
        # We'll expect it to be provided or calculated.
        minute = window_frame.get('_internal_minute', 15) 
        
        kills_blue = blue_team.get('kills', 0)
        kills_red = red_team.get('kills', 0)
        kills_diff = kills_blue - kills_red
        
        towers_blue = blue_team.get('towers', 0)
        towers_red = red_team.get('towers', 0)
        towers_diff = towers_blue - towers_red
        
        drakes_blue = len(blue_team.get('dragons', []))
        drakes_red = len(red_team.get('dragons', []))
        drakes_diff = drakes_blue - drakes_red
        
        gold_blue = blue_team.get('totalGold', 0)
        gold_red = red_team.get('totalGold', 0)
        gold_diff = gold_blue - gold_red
        
        # CS diff - needs details frame
        cs_blue = 0
        cs_red = 0
        if details_frame and 'participants' in details_frame:
            for p in details_frame['participants']:
                # participantId 1-5 = Blue, 6-10 = Red (usually)
                pid = p.get('participantId', 0)
                cs = p.get('creepScore', 0)
                if 1 <= pid <= 5:
                    cs_blue += cs
                elif 6 <= pid <= 10:
                    cs_red += cs
        
        cs_diff = cs_blue - cs_red
        
        # Metrics
        gold_diff_pct = (gold_diff / gold_red * 100) if gold_red > 0 else 0
        gold_per_min_blue = gold_blue / minute if minute > 0 else 0
        gold_per_min_red = gold_red / minute if minute > 0 else 0
        
        cs_per_min_blue = cs_blue / (minute + 1)
        cs_per_min_red = cs_red / (minute + 1)
        
        # Momentum
        kill_momentum_3min = int(kills_diff * 0.3)
        gold_momentum_5min = gold_diff * 0.2 / 5 if minute >= 5 else 0
        
        # Scores
        drake_control_score = drakes_diff * 1.5
        tower_sequence_score = towers_diff * 1.2
        baron_timing_advantage = -1 if (blue_team.get('barons', 0) + red_team.get('barons', 0)) == 0 else 2
        
        # Phases
        phase_early = 1 if minute < 15 else 0
        phase_mid = 1 if 15 <= minute < 25 else 0
        phase_late = 1 if minute >= 25 else 0
        
        # Advantages
        early_advantage = (cs_diff + gold_diff/10 + kills_diff*300) / minute if minute > 0 and phase_early else 0
        mid_advantage = (drakes_diff * 2 + towers_diff * 1.5) if phase_mid else 0
        late_advantage = gold_diff_pct + (blue_team.get('barons', 0) - red_team.get('barons', 0)) * 3 if phase_late else 0
        
        # Critical
        has_soul_point = 1 if (drakes_blue >= 3 or drakes_red >= 3) else 0
        gold_lead_critical = 1 if abs(gold_diff) > 3000 else 0
        tower_lead_critical = 1 if abs(towers_diff) >= 3 else 0
        
        # Interactions
        gold_diff_x_minute = gold_diff * minute
        kills_diff_x_minute = kills_diff * minute
        momentum_composite = kill_momentum_3min + gold_momentum_5min / 100
        
        features = {
            'kills_diff': kills_diff,
            'towers_diff': towers_diff,
            'drakes_diff': drakes_diff,
            'barons_BLUE': blue_team.get('barons', 0),
            'barons_RED': red_team.get('barons', 0),
            'gold_diff': gold_diff,
            'gold_diff_pct': gold_diff_pct,
            'cs_diff': cs_diff,
            'gold_per_min_blue': gold_per_min_blue,
            'gold_per_min_red': gold_per_min_red,
            'cs_per_min_blue': cs_per_min_blue,
            'cs_per_min_red': cs_per_min_red,
            'kill_momentum_3min': kill_momentum_3min,
            'gold_momentum_5min': gold_momentum_5min,
            'drake_control_score': drake_control_score,
            'tower_sequence_score': tower_sequence_score,
            'baron_timing_advantage': baron_timing_advantage,
            'early_advantage': early_advantage,
            'mid_advantage': mid_advantage,
            'late_advantage': late_advantage,
            'has_soul_point': has_soul_point,
            'gold_lead_critical': gold_lead_critical,
            'tower_lead_critical': tower_lead_critical,
            'phase_early': phase_early,
            'phase_mid': phase_mid,
            'phase_late': phase_late,
            'gold_diff_x_minute': gold_diff_x_minute,
            'kills_diff_x_minute': kills_diff_x_minute,
            'momentum_composite': momentum_composite
        }
        
        return features

    def get_minute_from_timestamp(self, rfc_ts: str, start_ts: str) -> float:
        """Helper to calculate match minute from Riot timestamps"""
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
            dt_now = datetime.strptime(rfc_ts, fmt)
            dt_start = datetime.strptime(start_ts, fmt)
            diff = dt_now - dt_start
            return diff.total_seconds() / 60
        except Exception as e:
            return 15.0
