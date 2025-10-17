"""
Feature builder for ML model input
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from database.db_manager import DatabaseManager
from logger.scrape_logger import ScrapeLogger


class FeatureBuilder:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = ScrapeLogger("FeatureBuilder")
    
    def build_ml_input(self, match_id: str) -> Optional[Dict]:
        """
        Build complete ML input row for a match
        
        Args:
            match_id: Match ID to build features for
            
        Returns:
            Dictionary with all features or None if data incomplete
        """
        self.logger.info(f"🔨 Building features for match {match_id}")
        
        # Get match data
        match = self.db.get_match(match_id)
        if not match:
            self.logger.error(f"Match {match_id} not found in database")
            return None
        
        # Get H2H data
        h2h = self.db.get_h2h(match_id)
        if not h2h:
            self.logger.warning(f"No H2H data for match {match_id}")
            return None
        
        # Initialize feature dict
        features = {
            'match_id': match_id,
            'event_id': match['event_id'],
            'date': match['date'],
            'team_home': match['team_home'],
            'team_away': match['team_away'],
            'score_home': match['score_home'],
            'score_away': match['score_away'],
            'label_home_win': 1 if match['score_home'] > match['score_away'] else 0
        }
        
        # Date features
        features['date_month'] = int(match['date'].strftime("%m"))
        features['date_day'] = int(match['date'].strftime("%w")) + 1
        
        # H2H between teams
        features['H2H_winrate_team1'] = h2h.get('home_win_rate')
        features['H2H_games'] = (h2h.get('wins_home', 0) + h2h.get('wins_away', 0))
        
        # Get team IDs from H2H
        home_team_id = h2h.get('home_team_id')
        away_team_id = h2h.get('away_team_id')
        
        if not home_team_id or not away_team_id:
            self.logger.error(f"Missing team IDs for match {match_id}")
            return None
        
        # Build features for each team
        for side, team_id in [('home', home_team_id), ('away', away_team_id)]:
            team_features = self._build_team_features(
                team_id, match['date'], match_id
            )
            if team_features:
                features.update({f"{side}_{k}": v for k, v in team_features.items()})
        
        # Difference features
        if 'home_last_3_winrate' in features and 'away_last_3_winrate' in features:
            features['diff_last_3_winrate'] = (
                features['home_last_3_winrate'] - features['away_last_3_winrate']
            )
        
        self.logger.info(f"✅ Features built for match {match_id}")
        return features
    
    def _build_team_features(self, team_id: str, match_date: datetime, 
                            match_id: str) -> Optional[Dict]:
        """Build features for a single team"""
        features = {}
        
        # Get team history
        history = self.db.get_team_history(team_id, before_date=match_date, limit=100)
        
        if history.empty or len(history) < 3:
            self.logger.warning(f"Insufficient history for team {team_id}")
            return None
        
        # Rolling winrate features
        last_3 = history.head(3)
        last_5 = history.head(5)
        
        features['last_3_winrate'] = (last_3['result'] == 'win').mean()
        features['last_5_winrate'] = (last_5['result'] == 'win').mean() if len(last_5) >= 5 else None
        
        # Score features
        features['last_3_avg_score_for'] = last_3['score_for'].mean()
        features['last_3_avg_score_against'] = last_3['score_against'].mean()
        
        # Current streak
        streak = 0
        if len(history) > 0:
            last_result = history.iloc[0]['result']
            for _, match in history.iterrows():
                if match['result'] == last_result:
                    streak += 1
                else:
                    break
            if last_result == 'loss':
                streak *= -1
        features['current_streak'] = streak
        
        # Get rankings
        rankings = self.db.get_rankings(before_date=match_date)
        team_name = history.iloc[0]['team_id']  # You'll need to get actual team name
        
        team_rankings = rankings[rankings['team_name'] == team_name]
        if not team_rankings.empty:
            team_rankings = team_rankings.sort_values('date', ascending=False)
            features['current_rank'] = team_rankings.iloc[0]['rank']
            if len(team_rankings) > 1:
                features['rank_change'] = (
                    team_rankings.iloc[1]['rank'] - team_rankings.iloc[0]['rank']
                )
        
        return features
    
    def calculate_historical_h2h_stats(self, team_id: str, match_date: datetime,
                                       n_matches: int = 3) -> Dict:
        """
        Calculate average H2H stats from team's last N matches
        
        Args:
            team_id: Team ID
            match_date: Match date (to filter history before this date)
            n_matches: Number of historical matches to analyze
            
        Returns:
            Dictionary with aggregated H2H statistics
        """
        history = self.db.get_team_history(team_id, before_date=match_date, limit=n_matches)
        
        if history.empty:
            return {}
        
        stats = {
            'n_matches_scraped': len(history)
        }
        
        # For each historical match, get its H2H data
        h2h_total = []
        h2h_wins = []
        ratings = []
        rating_stds = []
        adrs = []
        adr_stds = []
        swings = []
        swing_stds = []
        maps_total = []
        maps_wins = []
        maps_picked = []
        maps_avg_score_diffs = []
        
        for _, match_row in history.iterrows():
            hist_match_id = match_row['match_id']
            h2h_data = self.db.get_h2h(hist_match_id)
            
            if not h2h_data:
                continue
            
            # Determine if team was home or away
            if str(h2h_data['home_team_id']) == str(team_id):
                h2h_total.append(h2h_data.get('total_non_overtime', 0))
                h2h_wins.append(h2h_data.get('wins_home', 0))
                ratings.append(h2h_data.get('home_team_avg_rating'))
                rating_stds.append(h2h_data.get('home_team_std_rating'))
                adrs.append(h2h_data.get('home_team_avg_ADR'))
                adr_stds.append(h2h_data.get('home_team_std_ADR'))
                swings.append(h2h_data.get('home_team_avg_Swing'))
                swing_stds.append(h2h_data.get('home_team_std_Swing'))
                maps_total.append(h2h_data.get('maps_played', 0))
                maps_wins.append(h2h_data.get('home_maps_won', 0))
                maps_picked.append(h2h_data.get('home_maps_picked', 0))
                maps_avg_score_diffs.append(h2h_data.get('map_avg_score_diff', 0))
            elif str(h2h_data['away_team_id']) == str(team_id):
                h2h_total.append(h2h_data.get('total_non_overtime', 0))
                h2h_wins.append(h2h_data.get('wins_away', 0))
                ratings.append(h2h_data.get('away_team_avg_rating'))
                rating_stds.append(h2h_data.get('away_team_std_rating'))
                adrs.append(h2h_data.get('away_team_avg_ADR'))
                adr_stds.append(h2h_data.get('away_team_std_ADR'))
                swings.append(h2h_data.get('away_team_avg_Swing'))
                swing_stds.append(h2h_data.get('away_team_std_Swing'))
                maps_total.append(h2h_data.get('maps_played', 0))
                maps_wins.append(h2h_data.get('away_maps_won', 0))
                maps_picked.append(h2h_data.get('maps_played', 0) - h2h_data.get('home_maps_picked', 0))
                maps_avg_score_diffs.append(-h2h_data.get('map_avg_score_diff', 0))
        
        # Aggregate statistics
        stats['avg_h2h_winrate'] = np.sum(h2h_wins) / np.sum(h2h_total) if h2h_total else None
        stats['avg_rating'] = np.mean([r for r in ratings if r]) if ratings else None
        stats['avg_rating_std'] = np.mean([r for r in rating_stds if r]) if rating_stds else None
        stats['avg_adr'] = np.mean([a for a in adrs if a]) if adrs else None
        stats['avg_adr_std'] = np.mean([a for a in adr_stds if a]) if adr_stds else None
        stats['avg_swing'] = np.mean([s for s in swings if s]) if swings else None
        stats['avg_swing_std'] = np.mean([s for s in swing_stds if s]) if swing_stds else None
        stats['avg_maps_total'] = np.mean(maps_total) if maps_total else None
        stats['avg_map_winrate'] = np.sum(maps_wins) / np.sum(maps_total) if maps_total else None
        stats['avg_map_pickrate'] = np.sum(maps_picked) / np.sum(maps_total) if maps_total else None
        stats['avg_map_score_diff'] = np.mean(maps_avg_score_diffs) if maps_avg_score_diffs else None
        
        return stats