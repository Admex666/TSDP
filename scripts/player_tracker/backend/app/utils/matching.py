"""
Player matching utilities for data import
"""
from fuzzywuzzy import fuzz
from typing import List, Optional, Tuple
from app.models import Player
from config import config


def normalize_name(name: str) -> str:
    """
    Normalize player name for matching
    """
    # Remove accents, convert to lowercase, strip whitespace
    normalized = name.lower().strip()
    
    # Remove common suffixes
    suffixes = [" jr.", " sr.", " ii", " iii", " iv"]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    return normalized


def match_player(
    import_name: str,
    import_team: str,
    db_players: List[Player],
    threshold: int = None
) -> Tuple[Optional[Player], float]:
    """
    Find best matching player using fuzzy matching
    
    Args:
        import_name: Player name from import data
        import_team: Team name from import data
        db_players: List of players from database
        threshold: Minimum score to consider a match (default from config)
    
    Returns:
        Tuple of (matched_player, confidence_score)
    """
    if threshold is None:
        threshold = config.PLAYER_MATCH_THRESHOLD
    
    normalized_import_name = normalize_name(import_name)
    normalized_import_team = import_team.lower().strip() if import_team else ""
    
    best_match = None
    best_score = 0
    
    for player in db_players:
        # Name matching
        normalized_db_name = normalize_name(player.name)
        name_score = fuzz.ratio(normalized_import_name, normalized_db_name)
        
        # Team matching (if available)
        team_score = 0
        if player.team and import_team:
            normalized_db_team = player.team.name.lower().strip()
            team_score = fuzz.ratio(normalized_import_team, normalized_db_team)
        
        # Combined score (70% name, 30% team)
        if import_team and player.team:
            combined_score = 0.7 * name_score + 0.3 * team_score
        else:
            combined_score = name_score
        
        if combined_score > best_score and combined_score >= threshold:
            best_score = combined_score
            best_match = player
    
    return best_match, best_score


def match_team(
    import_team: str,
    db_teams: List,
    threshold: int = 75
) -> Tuple[Optional[any], float]:
    """
    Find best matching team using fuzzy matching
    
    Args:
        import_team: Team name from import data
        db_teams: List of teams from database
        threshold: Minimum score to consider a match
    
    Returns:
        Tuple of (matched_team, confidence_score)
    """
    normalized_import = import_team.lower().strip()
    
    best_match = None
    best_score = 0
    
    for team in db_teams:
        normalized_db = team.name.lower().strip()
        score = fuzz.ratio(normalized_import, normalized_db)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = team
    
    return best_match, best_score


def extract_position_group(detailed_position: str) -> str:
    """
    Map detailed position to position group (GK, DEF, MID, ATT)
    
    Args:
        detailed_position: Detailed position string (e.g., "CB", "LW", "CM")
    
    Returns:
        Position group string
    """
    detailed_position = detailed_position.upper().strip()
    
    for group, positions in config.POSITION_GROUPS.items():
        if detailed_position in [p.upper() for p in positions]:
            return group
    
    # Default mapping based on common patterns
    if "GK" in detailed_position or "KEEPER" in detailed_position:
        return "GK"
    elif any(x in detailed_position for x in ["DF", "CB", "LB", "RB", "WB", "BACK"]):
        return "DEF"
    elif any(x in detailed_position for x in ["FW", "ST", "CF", "WING", "FORWARD"]):
        return "ATT"
    else:
        return "MID"  # Default to midfielder
