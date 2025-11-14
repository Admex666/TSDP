"""
Value Betting Engine
Combines ML predictions with bookmaker odds to identify value bets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueBettingEngine:
    """
    Identifies value bets by comparing model predictions with bookmaker odds
    """
    
    def __init__(self, gb_model, rf_model, scaler, 
                 min_edge: float = 0.05, 
                 min_confidence: float = 0.55):
        """
        Args:
            gb_model: Trained Gradient Boosting model
            rf_model: Trained Random Forest model
            scaler: Fitted scaler for features
            min_edge: Minimum edge required to consider a value bet (default 5%)
            min_confidence: Minimum prediction confidence (default 55%)
        """
        self.gb_model = gb_model
        self.rf_model = rf_model
        self.scaler = scaler
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        
        self.feature_cols = [
            'kills_diff', 'towers_diff', 'drakes_diff', 'barons_BLUE', 'barons_RED',
            'gold_diff', 'gold_diff_pct', 'cs_diff', 'gold_per_min_blue', 'gold_per_min_red',
            'cs_per_min_blue', 'cs_per_min_red', 'kill_momentum_3min', 'gold_momentum_5min', 
            'drake_control_score', 'tower_sequence_score', 'baron_timing_advantage', 
            'early_advantage', 'mid_advantage', 'late_advantage', 'has_soul_point', 
            'gold_lead_critical', 'tower_lead_critical', 'phase_early', 'phase_mid',
            'phase_late', 'gold_diff_x_minute', 'kills_diff_x_minute', 'momentum_composite'
        ]
    
    def calculate_features(self, match_stats: Dict) -> Dict:
        """
        Calculate ML features from scraped match statistics
        
        Args:
            match_stats: Output from MatchStatsScraper
            
        Returns:
            Dictionary of feature values
        """
        blue = match_stats['blue_team']
        red = match_stats['red_team']
        
        # Parse game time (format: "29:00")
        game_time = match_stats['game_time']
        try:
            minutes, seconds = map(int, game_time.split(':'))
            minute = minutes + seconds / 60
        except:
            minute = 15  # Default fallback
        
        # Basic differences
        kills_diff = blue['kills'] - red['kills']
        towers_diff = blue['towers'] - red['towers']
        drakes_diff = len(blue['dragons']) - len(red['dragons'])
        gold_diff = blue['gold'] - red['gold']
        
        # Calculate CS from players
        cs_blue = sum(p['cs'] for p in match_stats['players'][:5])
        cs_red = sum(p['cs'] for p in match_stats['players'][5:])
        cs_diff = cs_blue - cs_red
        
        # Gold metrics
        gold_diff_pct = (gold_diff / red['gold'] * 100) if red['gold'] > 0 else 0
        gold_per_min_blue = blue['gold'] / minute if minute > 0 else 0
        gold_per_min_red = red['gold'] / minute if minute > 0 else 0
        
        # CS metrics
        cs_per_min_blue = cs_blue / (minute + 1)
        cs_per_min_red = cs_red / (minute + 1)
        
        # Momentum indicators
        kill_momentum_3min = int(kills_diff * 0.3)
        gold_momentum_5min = gold_diff * 0.2 / 5 if minute >= 5 else 0
        
        # Objective control scores
        drake_control_score = drakes_diff * 1.5
        tower_sequence_score = towers_diff * 1.2
        baron_timing_advantage = -1 if (blue['barons'] + red['barons']) == 0 else 2
        
        # Game phase indicators
        phase_early = 1 if minute < 15 else 0
        phase_mid = 1 if 15 <= minute < 25 else 0
        phase_late = 1 if minute >= 25 else 0
        
        # Phase-specific advantages
        early_advantage = (cs_diff + gold_diff/10 + kills_diff*300) / minute if minute > 0 and phase_early else 0
        mid_advantage = (drakes_diff * 2 + towers_diff * 1.5) if phase_mid else 0
        late_advantage = gold_diff_pct + (blue['barons'] - red['barons']) * 3 if phase_late else 0
        
        # Critical thresholds
        has_soul_point = 1 if (len(blue['dragons']) >= 3 or len(red['dragons']) >= 3) else 0
        gold_lead_critical = 1 if abs(gold_diff) > 3000 else 0
        tower_lead_critical = 1 if abs(towers_diff) >= 3 else 0
        
        # Interaction terms
        gold_diff_x_minute = gold_diff * minute
        kills_diff_x_minute = kills_diff * minute
        momentum_composite = kill_momentum_3min + gold_momentum_5min / 100
        
        features = {
            'kills_diff': kills_diff,
            'towers_diff': towers_diff,
            'drakes_diff': drakes_diff,
            'barons_BLUE': blue['barons'],
            'barons_RED': red['barons'],
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
    
    def predict_win_probability(self, features: Dict, use_ensemble: bool = True) -> Tuple[float, float]:
        """
        Predict win probabilities using ML models
        
        Args:
            features: Feature dictionary from calculate_features()
            use_ensemble: If True, average GB and RF predictions
            
        Returns:
            (prob_blue, prob_red) tuple
        """
        X_input = pd.DataFrame([features])[self.feature_cols]
        
        if use_ensemble:
            # Ensemble: average both models
            prob_blue_gb = self.gb_model.predict_proba(X_input)[0, 1]
            prob_blue_rf = self.rf_model.predict_proba(X_input)[0, 1]
            prob_blue = (prob_blue_gb + prob_blue_rf) / 2
        else:
            # Use only Gradient Boosting
            prob_blue = self.gb_model.predict_proba(X_input)[0, 1]
        
        prob_red = 1 - prob_blue
        return prob_blue, prob_red
    
    def calculate_value(self, predicted_prob: float, odds: float) -> float:
        """
        Calculate expected value / edge
        
        EV% = (predicted_prob * odds - 1) * 100
        
        Args:
            predicted_prob: Model's probability estimate (0-1)
            odds: Decimal odds from bookmaker
            
        Returns:
            Expected value as percentage (e.g., 5.0 means 5% edge)
        """
        ev = (predicted_prob * odds - 1) * 100
        return ev
    
    def find_value_bets(self, match_stats: Dict, odds_data: Dict, 
                    use_ensemble: bool = True, 
                    home_is_blue: bool = True) -> List[Dict]:  # ÚJ paraméter!
        """
        Identify value betting opportunities
        
        Args:
            match_stats: Output from MatchStatsScraper
            odds_data: Output from OddsScraper
            use_ensemble: Use ensemble prediction
            home_is_blue: If True, Tippmix "Hazai" = BLUE side; if False, "Hazai" = RED side
            
        Returns:
            List of value bet dictionaries
        """
        value_bets = []
        
        # Calculate features and get predictions
        features = self.calculate_features(match_stats)
        prob_blue, prob_red = self.predict_win_probability(features, use_ensemble)
        
        # Get game index from match stats
        game_index = match_stats.get('game_index', 1)
        
        logger.info(f"Model predictions (Game {game_index}) - Blue: {prob_blue:.1%}, Red: {prob_red:.1%}")
        logger.info(f"Team mapping: {'Hazai=BLUE, Vendég=RED' if home_is_blue else 'Hazai=RED, Vendég=BLUE'}")
        
        # Find relevant markets in odds data
        for market in odds_data['markets']:
            market_name = market['name']
            market_game_index = market.get('game_index', 0)
            
            # Only process markets for the current game
            if market_game_index != game_index:
                continue
            
            # Only process match winner markets
            if 'Ki nyeri' not in market_name and 'Winner' not in market_name:
                continue
            
            for option in market['options']:
                team = None
                predicted_prob = None
                
                # ÚJ LOGIKA: Determine team based on home_is_blue setting
                option_name = option['name']
                
                # Tippmix uses indices: typically options[0] = Hazai, options[1] = Vendég
                is_home_option = market['options'].index(option) == 0
                
                if home_is_blue:
                    # Standard mapping: Hazai = BLUE, Vendég = RED
                    if is_home_option:
                        team = 'BLUE'
                        predicted_prob = prob_blue
                    else:
                        team = 'RED'
                        predicted_prob = prob_red
                else:
                    # Inverted mapping: Hazai = RED, Vendég = BLUE
                    if is_home_option:
                        team = 'RED'
                        predicted_prob = prob_red
                    else:
                        team = 'BLUE'
                        predicted_prob = prob_blue
                
                if team is None or predicted_prob is None:
                    continue
                
                odds = option['odds']
                implied_prob = 1 / odds
                edge = self.calculate_value(predicted_prob, odds)
                
                # Check if this is a value bet
                if edge >= self.min_edge * 100 and predicted_prob >= self.min_confidence:
                    kelly_fraction = (predicted_prob * odds - 1) / (odds - 1)
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))
                    
                    # Confidence level
                    if edge >= 15:
                        confidence = 'HIGH'
                    elif edge >= 8:
                        confidence = 'MEDIUM'
                    else:
                        confidence = 'LOW'
                    
                    value_bet = {
                        'team': team,
                        'team_name': option_name,  # Ez most a Tippmix neve (Hazai/Vendég)
                        'market_name': market_name,
                        'game_index': game_index,
                        'odds': odds,
                        'predicted_prob': predicted_prob,
                        'implied_prob': implied_prob,
                        'edge': edge,
                        'kelly_fraction': kelly_fraction,
                        'confidence': confidence,
                        'timestamp': match_stats['timestamp'],
                        'game_time': match_stats['game_time']
                    }
                    
                    value_bets.append(value_bet)
                    logger.info(f"🎯 VALUE BET (Game {game_index}): {option_name} ({team} side) at {odds:.2f} (Edge: {edge:.1f}%)")
            
        return value_bets
    
    def should_cashout(self, current_prob: float, entry_prob: float, 
                       entry_odds: float, current_odds: float,
                       profit_threshold: float = 0.20) -> Tuple[bool, str]:
        """
        Determine if position should be cashed out
        
        Args:
            current_prob: Current model probability
            entry_prob: Entry probability when bet was placed
            entry_odds: Odds when bet was placed
            current_odds: Current cashout odds
            profit_threshold: Minimum profit % to consider cashout (default 20%)
            
        Returns:
            (should_cashout, reason) tuple
        """
        # Calculate probability shift
        prob_shift = current_prob - entry_prob
        
        # Calculate potential profit
        entry_ev = (entry_prob * entry_odds - 1) * 100
        current_ev = (current_prob * current_odds - 1) * 100
        
        profit_pct = ((current_odds / entry_odds) - 1) * 100
        
        # Cashout scenarios
        if prob_shift < -0.15:  # Lost 15%+ probability
            return True, f"Probability dropped {abs(prob_shift):.1%} - cut losses"
        
        if profit_pct >= profit_threshold and current_ev < 5:
            return True, f"Secured {profit_pct:.0f}% profit, edge reduced to {current_ev:.1f}%"
        
        if current_prob < 0.40:  # Below 40% win probability
            return True, f"Win probability too low ({current_prob:.1%})"
        
        return False, "Hold position - still good value"


if __name__ == "__main__":
    # Example usage
    import joblib
    
    # Load models (adjust paths)
    gb_model = joblib.load("models/live_gb_model_20251031.joblib")
    rf_model = joblib.load("models/live_rf_model_20251031.joblib")
    scaler = joblib.load("models/live_scaler_20251031.joblib")
    
    engine = ValueBettingEngine(gb_model, rf_model, scaler)
    
    # Test with sample data
    sample_stats = {
        'timestamp': datetime.now().isoformat(),
        'game_time': '29:00',
        'blue_team': {'kills': 15, 'towers': 1, 'inhibitors': 0, 
                      'barons': 0, 'gold': 51567, 'dragons': []},
        'red_team': {'kills': 34, 'towers': 8, 'inhibitors': 1,
                     'barons': 1, 'gold': 61768, 'dragons': []},
        'players': [{'cs': 191} for _ in range(10)]  # Simplified
    }
    
    features = engine.calculate_features(sample_stats)
    prob_blue, prob_red = engine.predict_win_probability(features)
    
    print(f"Blue win prob: {prob_blue:.1%}")
    print(f"Red win prob: {prob_red:.1%}")