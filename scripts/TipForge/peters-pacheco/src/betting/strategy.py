from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Bet:
    match_id: str
    selection: str # 'home', 'draw', 'away'
    stake: float
    odds: float
    model_prob: float
    implied_prob: float
    edge: float

class ValueBetEngine:
    """
    Identifies value bets based on model probabilities and market odds.
    """
    
    def __init__(self, margin_threshold: float = 0.05, stake: float = 1.0):
        self.margin_threshold = margin_threshold
        self.stake = stake
        
    def find_bets(self, match_id: str, probs: Dict[str, float], odds: Dict[str, float]) -> Optional[Bet]:
        """
        Identify if there is a value bet on any outcome.
        Returns the best bet (highest edge) or None.
        
        Args:
            probs: {'home_win': p, 'draw': p, 'away_win': p}
            odds: {'home_win': o, 'draw': o, 'away_win': o} (Decimal odds)
        """
        best_bet = None
        max_edge = -1.0
        
        outcomes = [('home_win', 'home'), ('draw', 'draw'), ('away_win', 'away')]
        
        for prob_key, selection in outcomes:
            model_p = probs.get(prob_key, 0.0)
            market_o = odds.get(prob_key, 0.0)
            
            if market_o <= 1.0:
                continue
                
            implied_p = 1.0 / market_o
            
            # Edge definition: Model Prob > Implied Prob * (1 + Margin)
            # OR Edge = (Model Prob * Odds) - 1 > Threshold?
            # Prompt says: model_probability > implied_market_probability * (1 + margin)
            
            required_prob = implied_p * (1.0 + self.margin_threshold)
            
            if model_p > required_prob:
                edge = model_p - implied_p # Simple edge metric
                if edge > max_edge:
                    max_edge = edge
                    best_bet = Bet(
                        match_id=match_id,
                        selection=selection,
                        stake=self.stake,
                        odds=market_o,
                        model_prob=model_p,
                        implied_prob=implied_p,
                        edge=edge
                    )
                    
        return best_bet
