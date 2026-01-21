"""
Metrics calculator for deriving advanced statistics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.leagues import (
    HIGH_QUALITY_XG_PER_SHOT, 
    SPECULATIVE_SHOOTING_PCT,
    PENALTY_BOX_X,
    SIX_YARD_BOX_X,
    CENTRAL_ZONE_Y_MIN,
    CENTRAL_ZONE_Y_MAX
)


class MetricsCalculator:
    """Calculate derived metrics and KPIs"""
    
    @staticmethod
    def calculate_per_90(value: float, matches: int) -> float:
        """Calculate per 90 minutes metric"""
        if matches == 0:
            return 0.0
        return round(value / matches, 2)
    
    @staticmethod
    def calculate_percentage(part: float, total: float) -> float:
        """Calculate percentage"""
        if total == 0:
            return 0.0
        return round((part / total) * 100, 1)
    
    @staticmethod
    def calculate_efficiency_metrics(stats: Dict) -> Dict:
        """Calculate shooting and passing efficiency metrics"""
        matches = stats.get('matches', 1)
        
        # Shooting efficiency
        shots = stats.get('shots', 0)
        shots_on_target = stats.get('shotsOnTarget', 0)
        goals = stats.get('goalsScored', 0)
        big_chances = stats.get('bigChances', 0)
        
        shooting_accuracy = MetricsCalculator.calculate_percentage(shots_on_target, shots)
        conversion_rate = MetricsCalculator.calculate_percentage(goals, shots)
        big_chance_conversion = MetricsCalculator.calculate_percentage(goals, big_chances)
        
        # Passing efficiency
        total_passes = stats.get('totalPasses', 0)
        accurate_passes = stats.get('accuratePasses', 0)
        pass_accuracy = MetricsCalculator.calculate_percentage(accurate_passes, total_passes)
        
        # Per 90 metrics
        goals_per_90 = MetricsCalculator.calculate_per_90(goals, matches)
        shots_per_90 = MetricsCalculator.calculate_per_90(shots, matches)
        
        return {
            'shooting_accuracy': shooting_accuracy,
            'conversion_rate': conversion_rate,
            'big_chance_conversion': big_chance_conversion,
            'pass_accuracy': pass_accuracy,
            'goals_per_90': goals_per_90,
            'shots_per_90': shots_per_90
        }
    
    @staticmethod
    def calculate_xg_metrics(stats: Dict, shotmap_df: pd.DataFrame = None) -> Dict:
        """Calculate xG-based metrics"""
        matches = stats.get('matches', 1)
        goals = stats.get('goalsScored', 0)
        shots = stats.get('shots', 0)
        
        # If we have shotmap data, calculate xG from it
        if shotmap_df is not None and not shotmap_df.empty and 'xG' in shotmap_df.columns:
            total_xg = shotmap_df['xG'].sum()
            xg_per_shot = total_xg / len(shotmap_df) if len(shotmap_df) > 0 else 0
        else:
            # Estimate xG (this would ideally come from API)
            # Using a rough estimate: big chances * 0.35 + other shots * 0.08
            big_chances = stats.get('bigChances', 0)
            other_shots = shots - big_chances
            total_xg = (big_chances * 0.35) + (other_shots * 0.08)
            xg_per_shot = total_xg / shots if shots > 0 else 0
        
        xg_per_90 = MetricsCalculator.calculate_per_90(total_xg, matches)
        xg_overperformance = ((goals - total_xg) / total_xg * 100) if total_xg > 0 else 0
        
        # Shot quality assessment
        shot_quality = "High" if xg_per_shot > HIGH_QUALITY_XG_PER_SHOT else "Medium" if xg_per_shot > 0.08 else "Low"
        
        return {
            'total_xg': round(total_xg, 2),
            'xg_per_90': round(xg_per_90, 2),
            'xg_per_shot': round(xg_per_shot, 3),
            'xg_overperformance': round(xg_overperformance, 1),
            'shot_quality': shot_quality,
            'finishing_performance': 'Overperform' if xg_overperformance > 5 else 'Underperform' if xg_overperformance < -5 else 'Expected'
        }
    
    @staticmethod
    def calculate_shot_location_metrics(shotmap_df: pd.DataFrame) -> Dict:
        """Calculate shot location breakdown from shotmap data"""
        if shotmap_df.empty:
            return {
                'inside_box_pct': 0,
                'outside_box_pct': 0,
                'six_yard_pct': 0,
                'central_pct': 0,
                'left_wing_pct': 0,
                'right_wing_pct': 0,
                'right_foot_pct': 0,
                'left_foot_pct': 0,
                'header_pct': 0
            }
        
        total_shots = len(shotmap_df)
        
        # Location metrics
        inside_box = len(shotmap_df[shotmap_df['playerX'] > PENALTY_BOX_X])
        outside_box = total_shots - inside_box
        six_yard = len(shotmap_df[shotmap_df['playerX'] > SIX_YARD_BOX_X])
        
        # Zone metrics
        central = len(shotmap_df[
            (shotmap_df['playerY'] > CENTRAL_ZONE_Y_MIN) & 
            (shotmap_df['playerY'] < CENTRAL_ZONE_Y_MAX)
        ])
        left_wing = len(shotmap_df[shotmap_df['playerY'] <= CENTRAL_ZONE_Y_MIN])
        right_wing = len(shotmap_df[shotmap_df['playerY'] >= CENTRAL_ZONE_Y_MAX])
        
        # Body part metrics
        right_foot = len(shotmap_df[shotmap_df['bodyPart'] == 'right-foot'])
        left_foot = len(shotmap_df[shotmap_df['bodyPart'] == 'left-foot'])
        header = len(shotmap_df[shotmap_df['bodyPart'] == 'head'])
        
        return {
            'inside_box_pct': MetricsCalculator.calculate_percentage(inside_box, total_shots),
            'outside_box_pct': MetricsCalculator.calculate_percentage(outside_box, total_shots),
            'six_yard_pct': MetricsCalculator.calculate_percentage(six_yard, total_shots),
            'central_pct': MetricsCalculator.calculate_percentage(central, total_shots),
            'left_wing_pct': MetricsCalculator.calculate_percentage(left_wing, total_shots),
            'right_wing_pct': MetricsCalculator.calculate_percentage(right_wing, total_shots),
            'right_foot_pct': MetricsCalculator.calculate_percentage(right_foot, total_shots),
            'left_foot_pct': MetricsCalculator.calculate_percentage(left_foot, total_shots),
            'header_pct': MetricsCalculator.calculate_percentage(header, total_shots)
        }
    
    @staticmethod
    def calculate_style_indicators(stats: Dict, shotmap_df: pd.DataFrame = None) -> Dict:
        """Calculate playing style indicators"""
        total_passes = stats.get('totalPasses', 1)
        possession = stats.get('averageBallPossession', 0)
        long_balls = stats.get('totalLongBalls', 0)
        crosses = stats.get('totalCrosses', 0)
        shots = stats.get('shots', 1)
        shots_outside_box = stats.get('shotsFromOutsideTheBox', 0)
        
        # Calculate percentages
        long_ball_pct = MetricsCalculator.calculate_percentage(long_balls, total_passes)
        cross_pct = MetricsCalculator.calculate_percentage(crosses, total_passes)
        shots_outside_pct = MetricsCalculator.calculate_percentage(shots_outside_box, shots)
        
        # Style classifications
        possession_style = "High" if possession > 55 else "Medium" if possession > 45 else "Low"
        direct_play = long_ball_pct > 10
        wing_oriented = cross_pct > 5
        speculative_shooting = shots_outside_pct > SPECULATIVE_SHOOTING_PCT
        
        # Get xG per shot if shotmap available
        xg_per_shot = 0
        if shotmap_df is not None and not shotmap_df.empty and 'xG' in shotmap_df.columns:
            xg_per_shot = shotmap_df['xG'].sum() / len(shotmap_df)
        
        high_quality_shots = xg_per_shot > HIGH_QUALITY_XG_PER_SHOT
        
        return {
            'possession_style': possession_style,
            'possession_pct': round(possession, 1),
            'direct_play': direct_play,
            'wing_oriented': wing_oriented,
            'speculative_shooting': speculative_shooting,
            'high_quality_shots': high_quality_shots,
            'long_ball_pct': round(long_ball_pct, 1),
            'cross_pct': round(cross_pct, 1),
            'shots_outside_pct': round(shots_outside_pct, 1)
        }
    
    @staticmethod
    def calculate_defensive_metrics(stats: Dict) -> Dict:
        """Calculate defensive metrics"""
        matches = stats.get('matches', 1)
        
        goals_conceded = stats.get('goalsConceded', 0)
        clean_sheets = stats.get('cleanSheets', 0)
        tackles = stats.get('tackles', 0)
        interceptions = stats.get('interceptions', 0)
        duels_won_pct = stats.get('duelsWonPercentage', 0)
        aerial_duels_won_pct = stats.get('aerialDuelsWonPercentage', 0)
        ground_duels_won_pct = stats.get('groundDuelsWonPercentage', 0)
        
        goals_conceded_per_90 = MetricsCalculator.calculate_per_90(goals_conceded, matches)
        clean_sheet_pct = MetricsCalculator.calculate_percentage(clean_sheets, matches)
        tackles_per_90 = MetricsCalculator.calculate_per_90(tackles, matches)
        interceptions_per_90 = MetricsCalculator.calculate_per_90(interceptions, matches)
        
        return {
            'goals_conceded_per_90': round(goals_conceded_per_90, 2),
            'clean_sheet_pct': round(clean_sheet_pct, 1),
            'tackles_per_90': round(tackles_per_90, 1),
            'interceptions_per_90': round(interceptions_per_90, 1),
            'duels_won_pct': round(duels_won_pct, 1),
            'aerial_duels_won_pct': round(aerial_duels_won_pct, 1),
            'ground_duels_won_pct': round(ground_duels_won_pct, 1)
        }
    
    @staticmethod
    def calculate_form_metrics(form_matches: List[Dict], team_id: int, tournament_id: int = None) -> Dict:
        """Calculate form metrics from recent matches"""
        if not form_matches:
            return {
                'points': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'form_string': ''
            }
        
        # Filter matches if tournament_id is provided
        filtered_matches = []
        for match in form_matches:
            # Only include finished matches
            if match.get('status', {}).get('type') != 'finished':
                continue
            
            # If tournament_id specified, only include matches from that tournament
            if tournament_id is not None:
                match_tournament_id = match.get('tournament', {}).get('uniqueTournament', {}).get('id')
                if match_tournament_id != tournament_id:
                    continue
            
            filtered_matches.append(match)
            
            # Stop when we have 5 matches
            if len(filtered_matches) >= 5:
                break
        
        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        form_string = []
        
        for match in filtered_matches:  # Use filtered matches
            is_home = match['homeTeam']['id'] == team_id
            home_score = match['homeScore']['current']
            away_score = match['awayScore']['current']
            
            if is_home:
                team_score = home_score
                opp_score = away_score
            else:
                team_score = away_score
                opp_score = home_score
            
            goals_for += team_score
            goals_against += opp_score
            
            if team_score > opp_score:
                wins += 1
                points += 3
                form_string.append('W')
            elif team_score < opp_score:
                losses += 1
                form_string.append('L')
            else:
                draws += 1
                points += 1
                form_string.append('D')
        
        return {
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'form_string': '-'.join(form_string)
        }
    
    @staticmethod
    def calculate_set_piece_metrics(stats: Dict) -> Dict:
        """Calculate set piece metrics"""
        matches = stats.get('matches', 1)
        
        # Corners
        corners = stats.get('corners', 0)
        corners_against = stats.get('cornersAgainst', 0)
        
        # Free kicks
        free_kick_goals = stats.get('freeKickGoals', 0)
        free_kick_shots = stats.get('freeKickShots', 0)
        
        # Penalties
        penalty_goals = stats.get('penaltyGoals', 0)
        penalties_taken = stats.get('penaltiesTaken', 0)
        penalties_conceded = stats.get('penaltiesCommited', 0)
        
        corners_per_90 = MetricsCalculator.calculate_per_90(corners, matches)
        corners_against_per_90 = MetricsCalculator.calculate_per_90(corners_against, matches)
        
        fk_conversion = MetricsCalculator.calculate_percentage(free_kick_goals, free_kick_shots)
        penalty_conversion = MetricsCalculator.calculate_percentage(penalty_goals, penalties_taken)
        
        return {
            'corners_per_90': round(corners_per_90, 1),
            'corners_against_per_90': round(corners_against_per_90, 1),
            'free_kick_goals': free_kick_goals,
            'free_kick_shots': free_kick_shots,
            'fk_conversion': round(fk_conversion, 1),
            'penalty_goals': penalty_goals,
            'penalties_taken': penalties_taken,
            'penalty_conversion': round(penalty_conversion, 1),
            'penalties_conceded': penalties_conceded
        }
