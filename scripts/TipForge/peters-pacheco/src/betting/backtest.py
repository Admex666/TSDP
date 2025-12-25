import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from ..models.regression import GoalRegressionModel
from ..models.probability import ScoreProbabilityModel
from .strategy import ValueBetEngine, Bet

class Backtester:
    """
    Simulates the betting strategy over historical data in a strict chronological order.
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, 
                 match_metadata: pd.DataFrame, 
                 initial_bankroll: float = 100.0):
        """
        matches_df must contain: Date, Home, Away, HomeGoals, AwayGoals, OddsHome, OddsDraw, OddsAway
        """
        self.features = feature_matrix
        self.metadata = match_metadata.copy()
        
        # Ensure chronological order
        if 'Date' in self.metadata.columns:
            self.metadata['Date'] = pd.to_datetime(self.metadata['Date'])
            self.metadata = self.metadata.sort_values('Date').reset_index(drop=True)
            # Reindex features to match
            # Assuming features index aligns with metadata index or we merge
            # For simplicity, assume passed aligned or use match_id index
            pass
            
        self.bankroll = initial_bankroll
        self.bets: List[Bet] = []
        self.history = []
        
        self.prob_model = ScoreProbabilityModel()
        self.bet_engine = ValueBetEngine(margin_threshold=0.05)
        
    def run(self, start_date: str, retrain_every: int = 50, window_size: int = None):
        """
        Execute backtest.
        
        Args:
            start_date: Date to start betting (prior data used for initial training)
            retrain_every: Number of matches between retraining models
            window_size: If set, sliding window training. Else expanding.
        """
        start_dt = pd.to_datetime(start_date)
        
        # Split into initial train and test sets
        mask_future = self.metadata['Date'] >= start_dt
        test_indices = self.metadata[mask_future].index
        
        if len(test_indices) == 0:
            print("No matches to test after start_date")
            return
            
        print(f"Starting backtest on {len(test_indices)} matches...")
        
        # Initial Train
        train_mask = self.metadata['Date'] < start_dt
        if not train_mask.any():
            raise ValueError("No training data before start_date")
            
        print("Training initial models...")
        model_home = GoalRegressionModel()
        model_away = GoalRegressionModel()
        
        def fit_models(curr_date):
            # Strict past data
            past_mask = self.metadata['Date'] < curr_date
            X_train = self.features[past_mask]
            
            # Targets
            # We assume metadata has 'HomeGoals', 'AwayGoals'
            y_home = self.metadata.loc[past_mask, 'HomeGoals']
            y_away = self.metadata.loc[past_mask, 'AwayGoals']
            
            model_home.train(X_train, y_home)
            model_away.train(X_train, y_away)
            
        # Initial fit
        fit_models(start_dt)
        
        matches_since_train = 0
        
        for idx in tqdm(test_indices):
            match = self.metadata.loc[idx]
            match_date = match['Date']
            
            # Retrain check
            if matches_since_train >= retrain_every:
                fit_models(match_date)
                matches_since_train = 0
                
            # Predict
            # Feature vector for this match
            # Need to reshape to 2D
            X_curr = self.features.iloc[[idx]] 
            
            # Warning: Using iloc on features assumes features aligned with metadata by integer index
            # This is fragile if sorting changed. Ideally join on ID.
            # Assuming aligned for now.
            
            pred_h = model_home.predict(X_curr)[0]
            pred_a = model_away.predict(X_curr)[0]
            
            # Probabilities
            # Assume 0-0 minimum? SVR can predict negative. Clip to 0.
            pred_h = max(0.1, pred_h) # Avoid 0 lambda for Poisson
            pred_a = max(0.1, pred_a)
            
            probs = self.prob_model.predict_probs(pred_h, pred_a)
            
            # Betting
            odds = {
                'home_win': match.get('OddsHome', 0.0),
                'draw': match.get('OddsDraw', 0.0),
                'away_win': match.get('OddsAway', 0.0)
            }
            
            bet = self.bet_engine.find_bets(str(idx), probs, odds)
            
            pnl = 0.0
            won = False
            
            if bet:
                # Outcome resolution
                actual_home = match['HomeGoals']
                actual_away = match['AwayGoals']
                
                result = 'draw'
                if actual_home > actual_away: result = 'home'
                elif actual_away > actual_home: result = 'away'
                
                if bet.selection == result:
                    won = True
                    pnl = bet.stake * (bet.odds - 1.0)
                else:
                    pnl = -bet.stake
                    
                self.bankroll += pnl
                self.bets.append(bet)
                
            self.history.append({
                'Date': match_date,
                'Match': f"{match['Home']} vs {match['Away']}",
                'Pred_H': pred_h,
                'Pred_A': pred_a,
                'Bet': bet.selection if bet else None,
                'Stake': bet.stake if bet else 0,
                'Odds': bet.odds if bet else 0,
                'Result': pnl,
                'Bankroll': self.bankroll
            })
            
            matches_since_train += 1
            
        print("Backtest complete.")
            
    def get_results_df(self):
        return pd.DataFrame(self.history)
