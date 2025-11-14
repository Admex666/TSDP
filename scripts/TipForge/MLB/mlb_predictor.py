# mlb_predictor.py
import pandas as pd
import numpy as np
from mlb_config import THRESHOLD

class MLBPredictor:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features
    
    def prepare_features(self, home_stats, away_stats):
        """
        Feature előkészítése a predikciókhoz
        """
        features_dict = {
            'home_batting_avg': home_stats['avg'],
            'visiting_batting_avg': away_stats['avg'],
            'home_slugging': home_stats['slg'],
            'visiting_slugging': away_stats['slg'],
            'home_obp': home_stats['obp'],
            'visiting_obp': away_stats['obp']
        }
        
        # Ensure we have all required features in the correct order
        feature_df = pd.DataFrame([features_dict])
        
        # Check for missing features
        missing_features = set(self.features) - set(feature_df.columns)
        if missing_features:
            print(f"WARNING: Missing features: {missing_features}")
            for feat in missing_features:
                feature_df[feat] = 0
        
        # Return in the correct order
        return feature_df[self.features]
    
    def predict(self, home_stats, away_stats):
        """
        MLB mérkőzés predikció készítése
        """
        try:
            X_original = self.prepare_features(home_stats, away_stats)
            
            if X_original.empty:
                print("ERROR: No features prepared")
                return None, pd.DataFrame(), np.array([])
            
            X_scaled = self.scaler.transform(X_original)
            
            # Predict probabilities
            proba = self.model.predict_proba(X_scaled)[0]
            prediction_class = self.model.predict(X_scaled)[0]
            
            return {
                'away_prob': proba[0],  # Away team win probability
                'home_prob': proba[1],  # Home team win probability
                'prediction_class': prediction_class
            }, X_original, X_scaled
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, pd.DataFrame(), np.array([])
    
    def analyze_value(self, predictions, odds, threshold=THRESHOLD):
        """
        Value betting elemzése
        """
        if predictions is None or odds is None:
            return {'home_value': False, 'away_value': False}
        
        home_prob = predictions['home_prob']
        away_prob = predictions['away_prob']
        
        home_implied = 1 / odds['H_odds'] if 'H_odds' in odds else 0
        away_implied = 1 / odds['A_odds'] if 'A_odds' in odds else 0
        
        value_bets = {
            'home_value': home_prob > home_implied + threshold,
            'away_value': away_prob > away_implied + threshold
        }
        
        return value_bets
    
    def get_prediction_explanation(self, home_team, away_team, home_stats, away_stats, predictions):
        """
        Predikció magyarázatának generálása
        """
        explanation = f"""
📊 {home_team} vs {away_team} - STATISZTIKAI ELEMZÉS

🏠 HOME TEAM ({home_team}):
   • Batting Average: {home_stats['avg']:.3f}
   • Slugging: {home_stats['slg']:.3f}
   • On-Base %: {home_stats['obp']:.3f}

✈️ AWAY TEAM ({away_team}):
   • Batting Average: {away_stats['avg']:.3f}
   • Slugging: {away_stats['slg']:.3f}
   • On-Base %: {away_stats['obp']:.3f}

🎯 PREDICTION:
   • Home Win: {predictions['home_prob']:.1%}
   • Away Win: {predictions['away_prob']:.1%}
   • Predicted Winner: {'Home' if predictions['prediction_class'] == 1 else 'Away'}
        """
        
        return explanation.strip()