import numpy as np
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Optional

def calculate_no_vig_odds(odds_list: List[float]) -> List[float]:
    """
    Calculates no-vig (fair) odds from a list of bookmaker odds.
    Formula: No-Vig Odds = Odds / (1 - Margin)
    Margin = (Sum(1/Odds) - 1)
    """
    if not odds_list or any(o <= 1 for o in odds_list):
        return odds_list
        
    probabilities = [1/o for o in odds_list]
    overround = sum(probabilities)
    
    # Fair probabilities
    fair_probs = [p / overround for p in probabilities]
    
    # No-vig odds
    no_vig_odds = [1/p for p in fair_probs]
    return no_vig_odds

def find_best_match(target: str, candidates: List[str], threshold: int = 80) -> Optional[str]:
    """
    Finds the best matching string from a list of candidates using fuzzy matching.
    """
    if not target or not candidates:
        return None
        
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        # Check both partial and simple ratio
        score = max(fuzz.ratio(target.lower(), candidate.lower()), 
                    fuzz.partial_ratio(target.lower(), candidate.lower()))
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
            
    return best_match

def calculate_ev(prob: float, odds: float) -> float:
    """
    Calculates Expected Value (EV).
    EV = (Probability * Odds) - 1
    """
    return (prob * odds) - 1
