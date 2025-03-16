# predictor.py
import pandas as pd
import numpy as np
from datetime import datetime
from config import THRESHOLD

class MatchPredictor:
    def __init__(self, models, scaler, feature_columns):
        self.models = models
        self.scaler = scaler
        self.feature_columns = feature_columns
    
    def prepare_features(self, home_stats, away_stats, odds):
        """Feature vektor előkészítése"""
        home_implied = 1 / odds['home']
        draw_implied = 1 / odds['draw']
        away_implied = 1 / odds['away']
        
        features = {
            'Home_Implied_Prob': home_implied,
            'Draw_Implied_Prob': draw_implied,
            'Away_Implied_Prob': away_implied,
            'Home_Points_Last_5': home_stats['points'],
            'Home_Goals_For_Last_5': home_stats['goals_for'],
            'Home_Goals_Against_Last_5': home_stats['goals_against'],
            'Away_Points_Last_5': away_stats['points'],
            'Away_Goals_For_Last_5': away_stats['goals_for'],
            'Away_Goals_Against_Last_5': away_stats['goals_against'],
            'Home_Advantage': 1,
            'Days_Since_Last_Home_Match': home_stats['days_since'],
            'Days_Since_Last_Away_Match': away_stats['days_since']
        }
        
        return pd.DataFrame([features])[self.feature_columns]
    
    def predict(self, home_stats, away_stats, odds):
        """Predikció készítése"""
        X_new = self.prepare_features(home_stats, away_stats, odds)
        X_scaled = self.scaler.transform(X_new)
        
        probs = {}
        for model_name, model in self.models.items():
            probs[model_name] = model.predict_proba(X_scaled)[0]
        
        return probs
    
    def analyze_value(self, probs, odds, threshold=THRESHOLD):
        """Érték fogadások analízise"""
        home_implied = 1 / odds['home']
        draw_implied = 1 / odds['draw']
        away_implied = 1 / odds['away']
        
        value_bets = {}
        for model_name, prob in probs.items():
            value_bets[model_name] = {
                'home': prob[0] > home_implied + threshold,
                'draw': prob[1] > draw_implied + threshold,
                'away': prob[2] > away_implied + threshold
            }
        
        return value_bets