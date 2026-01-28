from typing import Dict, Any

def get_position_category(position: str) -> str:
    """Normalize position to G, D, M, F"""
    if not position: return "F" # Default fallback
    position = position.upper()
    if position in ['G', 'GK', 'GOALKEEPER']: return 'G'
    if position in ['D', 'DF', 'DEFENDER']: return 'G' # catch all defenders as D usually, but let's be strict
    if position in ['D', 'DF', 'DEFENDER']: return 'D'
    if position in ['M', 'MF', 'MIDFIELDER']: return 'M'
    if position in ['F', 'FW', 'FORWARD', 'ST']: return 'F'
    return 'F'

def calculate_rating_points(rating: float) -> int:
    """Calculates points based on SofaScore rating."""
    if not rating: return 0
    if rating >= 9.0: return 3
    if rating >= 8.0: return 2
    if rating >= 7.0: return 1
    if rating >= 6.5: return 0
    if rating >= 6.0: return -1
    return -2

def calculate_fantasy_points(stats: Dict[str, Any], position: str) -> float:
    """
    Calculate Fantasy Points based on Official SofaScore Rules.
    """
    points = 0.0
    pos = get_position_category(position)
    
    # 1. SofaScore Rating Points (-2 to 3)
    # We add this ON TOP of stats. "In addition to statistics"
    rating = stats.get('rating')
    if rating:
        points += calculate_rating_points(rating)

    # 2. Key Actions
    # --- Goals ---
    goals = stats.get('goals', 0)
    if pos in ['G', 'D']: points += goals * 6
    elif pos == 'M':      points += goals * 5
    elif pos == 'F':      points += goals * 4
        
    # --- Assists ---
    # API key is 'goalAssist' usually
    assists = stats.get('goalAssist', 0)
    if pos in ['G', 'D']: points += assists * 4
    elif pos in ['M', 'F']: points += assists * 3
    
    # --- Appearance ---
    minutes = stats.get('minutesPlayed', 0)
    if minutes >= 60:
        points += 2
    elif minutes > 0:
        points += 1
    
    # --- Clean Sheet (G, D only, 60+ mins) ---
    # We usually rely on goalsConceded being 0.
    goals_conceded = stats.get('goalsConceded', 0)
    if pos in ['G', 'D'] and minutes >= 60 and goals_conceded == 0:
        points += 4
        
    # --- Negative Steps ---
    # Goals Conceded (G, D only): -1 for every 2
    if pos in ['G', 'D']:
        points -= (goals_conceded // 2) * 1
        
    points -= stats.get('ownGoals', 0) * 2
    
    # Red Card 
    points -= stats.get('redCards', 0) * 3 
    
    # Yellow Cards
    points -= stats.get('yellowCards', 0) * 1
    
    # --- Penalties ---
    points += stats.get('penaltyWon', 0) * 2
    points -= stats.get('penaltyConceded', 0) * 2
    points -= stats.get('penaltyMissed', 0) * 3
    
    # --- Goalkeeper Specific ---
    if pos == 'G':
        # "Penalties saved" -> 5 pts.
        points += stats.get('penaltySave', 0) * 5
        
        # Saves (NOT penalties)
        # Inside box: 1 pt for every 2
        saves_in = stats.get('savedShotsFromInsideTheBox', 0)
        points += (saves_in // 2) * 1
        
        # Outside box: 1 pt for every 3
        # Derived: Total Saves - Inside Box
        total_saves = stats.get('saves', 0)
        saves_out = max(0, total_saves - saves_in)
        points += (saves_out // 3) * 1
        
        # Punches + High Claims (sum // 2)
        punches = stats.get('punches', 0)
        high_claims = stats.get('goodHighClaim', 0) 
        points += ((punches + high_claims) // 2) * 1
        
        # Successful runs out -> accurateKeeperSweeper
        points += stats.get('accurateKeeperSweeper', 0) * 1 

    # --- Outfield / General Stats ---
    # Long balls: 3 accurate (60% acc) -> 1 pt.
    acc_lb = stats.get('accurateLongBalls', 0)
    tot_lb = stats.get('totalLongBalls', 0)
    if acc_lb >= 3 and tot_lb > 0:
        if (acc_lb / tot_lb) >= 0.60:
            points += 1
    
    # Clearance off line: +2
    points += stats.get('clearanceOffLine', 0) * 2
            
    # Clearances: 1 per 5
    points += (stats.get('totalClearance', 0) // 5) * 1
    
    # Shots blocked: 1 per 2 
    if pos != 'G':
        points += (stats.get('blockedScoringAttempt', 0) // 2) * 1
    
    # Interceptions: 1 per 3
    points += (stats.get('interceptionWon', 0) // 3) * 1
    
    # Tackles won: 1 per 3
    points += (stats.get('wonTackle', 0) // 3) * 1
    
    # Duels won: 3+ (50% acc) -> 1 pt
    # Summing Ground + Aerial Duels
    duels_won = stats.get('duelWon', 0) + stats.get('aerialWon', 0)
    duels_lost = stats.get('duelLost', 0) + stats.get('aerialLost', 0)
    total_duels = duels_won + duels_lost
    if duels_won >= 3 and total_duels > 0:
        if (duels_won / total_duels) >= 0.50:
            points += 1
            
    # Passing: 40+ (90% acc) -> 1 pt
    acc_pass = stats.get('accuratePass', 0)
    tot_pass = stats.get('totalPass', 0) 
    if acc_pass >= 40 and tot_pass > 0:
        if (acc_pass / tot_pass) >= 0.90:
            points += 1
            
    # Key passes: 1 per 2
    # Note: Confirmed key is 'keyPass' (singular)
    points += (stats.get('keyPass', 0) // 2) * 1
    
    # Succ dribbles: 3+ (60% acc) -> 1 pt
    # Using 'wonContest' / 'totalContest' as proxy for dribbles is still the best guess based on available keys
    succ_drib = stats.get('wonContest', 0) 
    tot_drib = stats.get('totalContest', 0)
    if succ_drib >= 3 and tot_drib > 0:
        if (succ_drib / tot_drib) >= 0.60:
            points += 1
    
    # Dispossessed: -1 per 3
    points -= (stats.get('dispossessed', 0) // 3) * 1
    
    # Offsides: -1 per 2
    points -= (stats.get('totalOffside', 0) // 2) * 1 # Corrected to 'totalOffside' from columns list
    
    # Was fouled: 1 per 3
    points += (stats.get('wasFouled', 0) // 3) * 1

    return float(points)
