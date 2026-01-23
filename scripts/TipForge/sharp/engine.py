import logging
from typing import List, Dict, Tuple
from utils import find_best_match, calculate_no_vig_odds, calculate_ev

logger = logging.getLogger(__name__)

class ValueBetEngine:
    def __init__(self, min_ev: float = 0.0, match_threshold: int = 85):
        self.min_ev = min_ev
        self.match_threshold = match_threshold

    def find_value_bets(self, pinnacle_matches: List[Dict], tippmix_matches: List[Dict]) -> List[Dict]:
        """
        Pairs matches from both sources and identifies value bets.
        """
        value_bets = []
        
        # We'll use Pinnacle as the source of "truth" (sharp odds)
        for p_match in pinnacle_matches:
            p_home = p_match['home_team']
            p_away = p_match['away_team']
            p_odds = p_match['odds']
            
            # Calculate no-vig Pinnacle odds
            p_no_vig = calculate_no_vig_odds(p_odds)
            fair_probabilities = [1/o for o in p_no_vig]
            
            # Try to find the matching match in Tippmix
            # We match by concatenating team names to handle name order or partial matches
            p_full_name = f"{p_home} {p_away}"
            
            best_tippmix_match = None
            for t_match in tippmix_matches:
                t_home = t_match['home_team']
                t_away = t_match['away_team']
                t_full_name = f"{t_home} {t_away}"
                
                # Use fuzzy matching on the full string or individual teams
                # Matching home and away separately is safer
                home_match = find_best_match(p_home, [t_home], self.match_threshold)
                away_match = find_best_match(p_away, [t_away], self.match_threshold)
                
                if home_match and away_match:
                    best_tippmix_match = t_match
                    logger.info(f"✅ Paired Match: {p_home} vs {p_away} (Pinnacle) <-> {t_home} vs {t_away} (Tippmix)")
                    break

            
            if best_tippmix_match:
                t_odds = best_tippmix_match['odds']
                
                # Check if market sizes match (e.g., both 1X2 or both 12)
                if len(p_no_vig) == len(t_odds):
                    for i in range(len(t_odds)):
                        tippmix_odd = t_odds[i]
                        fair_prob = fair_probabilities[i]
                        
                        ev = calculate_ev(fair_prob, tippmix_odd)
                        
                        # Return ALL paired matches
                        outcome_name = ["Home", "Draw", "Away"][i] if len(t_odds) == 3 else ["Home", "Away"][i]
                        value_bets.append({
                            'match': f"{p_home} vs {p_away}",
                            'outcome': outcome_name,
                            'tippmix_odds': tippmix_odd,
                            'pinnacle_no_vig': p_no_vig[i],
                            'ev': ev,
                            'fair_prob': fair_prob
                        })
                else:
                    logger.warning(f"Market size mismatch for {p_full_name}: Pinnacle {len(p_no_vig)}, Tippmix {len(t_odds)}")

        return value_bets
