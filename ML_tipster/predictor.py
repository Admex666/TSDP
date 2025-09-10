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
        
        if hasattr(scaler, 'center_'):
            self.scaler_means_ = scaler.center_
            self.scaler_scales_ = scaler.scale_
        else:
            # StandardScaler vagy más scaler esetén
            self.scaler_means_ = getattr(scaler, 'mean_', None)
            self.scaler_scales_ = getattr(scaler, 'scale_', None)
    
    def prepare_features(self, home_stats, away_stats, odds):
        """Feature vektor előkészítése ORIGINAL értékekkel"""
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
        
        df_features = pd.DataFrame([features])
        missing_features = set(self.feature_columns) - set(df_features.columns)
        if missing_features:
            print(f"WARNING: Missing features: {missing_features}")
            # Hiányzó feature-öket nullával töltjük fel
            for feat in missing_features:
                df_features[feat] = 0
        
        return df_features[self.feature_columns]
    
    def predict(self, home_stats, away_stats, odds):
        """Predikció készítése"""
        X_original = self.prepare_features(home_stats, away_stats, odds)
        
        # Ellenőrizzük, hogy van-e adat
        if X_original.empty or len(X_original.columns) == 0:
            print("ERROR: No features prepared")
            return {}, pd.DataFrame(), np.array([])
        
        try:
            X_scaled = self.scaler.transform(X_original)
            
            probs = {}
            for model_name, model in self.models.items():
                try:
                    probs[model_name] = model.predict_proba(X_scaled)[0]
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    continue
            
            return probs, X_original, X_scaled
            
        except Exception as e:
            print(f"Error in scaling: {e}")
            return {}, X_original, np.array([])
    
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
    
    def explain_prediction(self, model_name, X_original, X_scaled, prediction):
        """Predikció részletes magyarázata ORIGINAL értékekkel"""
        try:
            if model_name not in self.models:
                return []
                
            model = self.models[model_name]
            
            if X_original.empty or len(X_scaled) == 0:
                return []
            
            if hasattr(model, 'feature_importances_'):
                contributions = model.feature_importances_
            elif hasattr(model, 'coef_'):
                contributions = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return []
                
            # Ellenőrizzük a dimenziók konzisztenciáját
            if len(contributions) != len(self.feature_columns) or len(X_scaled[0]) != len(self.feature_columns):
                print(f"Dimension mismatch: contributions={len(contributions)}, features={len(self.feature_columns)}, X_scaled={len(X_scaled[0])}")
                return []
            
            scaled_values = X_scaled[0]
            original_values = X_original.iloc[0].values
            
            impacts = []
            for i, feat_name in enumerate(self.feature_columns):
                if i < len(original_values) and i < len(scaled_values) and i < len(contributions):
                    impact_percent = contributions[i] * scaled_values[i] * 100
                    
                    impacts.append({
                        'feature': feat_name,
                        'original_value': original_values[i],
                        'scaled_value': scaled_values[i],
                        'impact': impact_percent
                    })
            
            impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
            return impacts
            
        except Exception as e:
            print(f"Explanation error for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return []