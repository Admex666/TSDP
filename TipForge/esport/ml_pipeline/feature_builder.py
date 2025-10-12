"""
Feature engineering: rolling statistics, H2H history.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from .ml_config import *

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """ML feature-ök építése."""
    
    def __init__(self):
        pass
    
    def build_ml_dataset(self) -> pd.DataFrame:
        """
        Teljes ML dataset építése:
        - Target matches (label)
        - Match details (H2H, player stats)
        - Rolling features (team history alapján)
        
        Returns:
            ML-ready DataFrame
        """
        logger.info("="*80)
        logger.info("🔧 ML DATASET ÉPÍTÉSE")
        logger.info("="*80)
        
        # 1. Betöltés
        target_matches = pd.read_csv(ML_TARGET_MATCHES_CSV)
        match_details = pd.read_csv(ML_MATCH_DETAILS_CSV)
        team_history = pd.read_csv(ML_TEAM_HISTORY_CSV)
        
        # Dátum konverziók
        target_matches['date'] = pd.to_datetime(target_matches['date'], errors='coerce')
        team_history['match_date'] = pd.to_datetime(team_history['match_date'], errors='coerce')
        
        logger.info(f"📊 Target matches: {len(target_matches)}")
        logger.info(f"📊 Match details: {len(match_details)}")
        logger.info(f"📊 Team history: {len(team_history)}")
        
        # 2. Join: target matches + match details
        ml_data = target_matches.merge(
            match_details,
            on='match_id',
            how='left',
            suffixes=('', '_detail')
        )
        
        logger.info(f"📊 Merged data: {len(ml_data)}")
        
        # 3. Rolling features minden meccshez
        logger.info("\n🔧 Rolling features számítása...")
        
        for idx, row in ml_data.iterrows():
            match_date = row['date']
            home_team_id = row.get('home_team_id')
            away_team_id = row.get('away_team_id')
            
            if pd.isna(match_date) or pd.isna(home_team_id) or pd.isna(away_team_id):
                continue
            
            # Home team rolling features
            home_features = self._compute_rolling_features(
                team_history, str(home_team_id), match_date
            )
            
            # Away team rolling features
            away_features = self._compute_rolling_features(
                team_history, str(away_team_id), match_date
            )
            
            # H2H history
            h2h_winrate = self._compute_h2h_winrate(
                team_history, str(home_team_id), str(away_team_id), match_date
            )
            
            # Feature-ök hozzáadása
            for key, value in home_features.items():
                ml_data.loc[idx, f'home_{key}'] = value
            
            for key, value in away_features.items():
                ml_data.loc[idx, f'away_{key}'] = value
            
            ml_data.loc[idx, 'h2h_home_winrate'] = h2h_winrate
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(ml_data)} matches")
        
        # 4. Label: home team nyert-e
        ml_data['label_home_win'] = (ml_data['score_home'] > ml_data['score_away']).astype(int)
        
        # 5. Mentés
        ml_data.to_csv(ML_DATASET_CSV, index=False)
        logger.info(f"\n✅ ML dataset mentve: {ML_DATASET_CSV} ({len(ml_data)} sor)")
        
        return ml_data
    
    def _compute_rolling_features(self, team_history: pd.DataFrame, team_id: str, 
                                   before_date: pd.Timestamp) -> Dict:
        """
        Rolling statistics egy csapatnak egy időpont ELŐTT.
        
        Args:
            team_history: Team history DataFrame
            team_id: Team ID
            before_date: Csak ezen dátum előtti meccsek
        
        Returns:
            Dictionary: last_3_winrate, last_5_winrate, last_3_avg_score, days_since_last, streak
        """
        # Csak ezen csapat meccseit, időrendben
        team_matches = team_history[
            (team_history['team_id'] == team_id) &
            (team_history['match_date'] < before_date)
        ].sort_values('match_date', ascending=False)
        
        if len(team_matches) == 0:
            return {
                'last_3_winrate': None,
                'last_5_winrate': None,
                'last_10_winrate': None,
                'last_3_avg_score_for': None,
                'last_3_avg_score_against': None,
                'days_since_last_match': None,
                'current_streak': 0
            }
        
        # Winrates
        last_3 = team_matches.head(3)
        last_5 = team_matches.head(5)
        last_10 = team_matches.head(10)
        
        last_3_wr = (last_3['result'] == 'win').mean() if len(last_3) > 0 else None
        last_5_wr = (last_5['result'] == 'win').mean() if len(last_5) > 0 else None
        last_10_wr = (last_10['result'] == 'win').mean() if len(last_10) > 0 else None
        
        # Avg scores
        last_3_avg_for = last_3['score_for'].mean() if len(last_3) > 0 else None
        last_3_avg_against = last_3['score_against'].mean() if len(last_3) > 0 else None
        
        # Days since last match
        last_match_date = team_matches.iloc[0]['match_date']
        days_since = (before_date - last_match_date).days if pd.notna(last_match_date) else None
        
        # Current streak
        streak = 0
        if len(team_matches) > 0:
            last_result = team_matches.iloc[0]['result']
            for _, match in team_matches.iterrows():
                if match['result'] == last_result:
                    streak += 1
                else:
                    break
            if last_result == 'loss':
                streak *= -1
        
        return {
            'last_3_winrate': round(last_3_wr, 4) if last_3_wr is not None else None,
            'last_5_winrate': round(last_5_wr, 4) if last_5_wr is not None else None,
            'last_10_winrate': round(last_10_wr, 4) if last_10_wr is not None else None,
            'last_3_avg_score_for': round(last_3_avg_for, 2) if last_3_avg_for is not None else None,
            'last_3_avg_score_against': round(last_3_avg_against, 2) if last_3_avg_against is not None else None,
            'days_since_last_match': days_since,
            'current_streak': streak
        }
    
    def _compute_h2h_winrate(self, team_history: pd.DataFrame, team_id: str, 
                            opponent_id: str, before_date: pd.Timestamp) -> float:
        """
        Head-to-head winrate két csapat között (team_id szempontjából).
        
        Args:
            team_history: Team history DataFrame
            team_id: Team ID
            opponent_id: Opponent team ID
            before_date: Csak ezen dátum előtti meccsek
        
        Returns:
            H2H winrate (vagy None ha nincs előzmény)
        """
        # Team_id meccseit az opponent ellen
        h2h_matches = team_history[
            (team_history['team_id'] == team_id) &
            (team_history['opponent_id'] == opponent_id) &
            (team_history['match_date'] < before_date)
        ]
        
        if len(h2h_matches) == 0:
            return None
        
        winrate = (h2h_matches['result'] == 'win').mean()
        return round(winrate, 4)