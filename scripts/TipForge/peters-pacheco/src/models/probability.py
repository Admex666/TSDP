import numpy as np
from scipy.stats import poisson

class ScoreProbabilityModel:
    """
    Converts predicted goals into match outcome probabilities using Poisson distribution.
    """
    
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        
    def predict_probs(self, home_goals_avg: float, away_goals_avg: float) -> dict:
        """
        Calculate Home Win, Draw, and Away Win probabilities.
        
        Args:
            home_goals_avg: Predicted expected goals for home team
            away_goals_avg: Predicted expected goals for away team
        
        Returns:
            dict: {'home_win': p, 'draw': p, 'away_win': p}
        """
        # Generate Poisson distributions
        # P(goals = k) = (lambda^k * e^-lambda) / k!
        
        h_probs = [poisson.pmf(i, home_goals_avg) for i in range(self.max_goals)]
        a_probs = [poisson.pmf(i, away_goals_avg) for i in range(self.max_goals)]
        
        # Calculate score matrix probability P(score = x-y) = P(H=x) * P(A=y) (assuming independence)
        
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for h in range(self.max_goals):
            for a in range(self.max_goals):
                p = h_probs[h] * a_probs[a]
                if h > a:
                    prob_home += p
                elif h == a:
                    prob_draw += p
                else:
                    prob_away += p
                    
        # Normalize (truncation might cause sum < 1)
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
            
        return {
            'home_win': prob_home,
            'draw': prob_draw,
            'away_win': prob_away
        }
