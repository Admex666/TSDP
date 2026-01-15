"""
ID-based player tracking service
Tracks players using FBref and SofaScore IDs
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session

from app.models import Player, Team, League, PlayerStats
from app.utils.matching import extract_position_group
from config import config

# Import scraper modules directly to avoid circular imports
try:
    from app.services.sofascore_scraper import scrape_sofascore
except ImportError:
    scrape_sofascore = None


class IDBasedTrackerService:
    """
    Service for tracking players using external IDs
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.errors = []
    
    def add_players_by_id(
        self,
        player_ids: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add players to tracking using FBref and/or SofaScore IDs
        
        Args:
            player_ids: List of dicts with format:
                {
                    "fbref_id": "abc123",  # optional
                    "sofascore_id": 12345,  # optional
                    "name": "Player Name"  # optional, for display
                }
        
        Returns:
            Summary dict with added players
        """
        added = []
        already_exists = []
        errors = []
        
        for player_data in player_ids:
            fbref_id = player_data.get('fbref_id')
            sofascore_id = player_data.get('sofascore_id')
            name = player_data.get('name', 'Unknown')
            
            if not fbref_id and not sofascore_id:
                errors.append(f"{name}: No ID provided")
                continue
            
            try:
                # Check if player already exists
                existing = None
                if fbref_id:
                    existing = self.db.query(Player).filter(
                        Player.fbref_id == fbref_id
                    ).first()
                
                if not existing and sofascore_id:
                    existing = self.db.query(Player).filter(
                        Player.sofascore_id == sofascore_id
                    ).first()
                
                if existing:
                    # Update IDs if missing
                    updated = False
                    if fbref_id and not existing.fbref_id:
                        existing.fbref_id = fbref_id
                        updated = True
                    if sofascore_id and not existing.sofascore_id:
                        existing.sofascore_id = sofascore_id
                        updated = True
                    
                    if not existing.tracked:
                        existing.tracked = True
                        added.append(existing.name)
                    else:
                        already_exists.append(existing.name)
                else:
                    # Create new player with default position if not provided
                    new_player = Player(
                        name=name,
                        position='MID',  # Default position
                        fbref_id=fbref_id,
                        sofascore_id=sofascore_id,
                        tracked=True
                    )
                    self.db.add(new_player)
                    self.db.flush()
                    added.append(name)
                    
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                continue
        
        self.db.commit()
        
        return {
            "status": "success" if len(errors) == 0 else "partial",
            "added": added,
            "already_tracked": already_exists,
            "errors": errors
        }
    
    def fetch_tracked_players_data(
        self,
        source: str = "both"
    ) -> Dict[str, Any]:
        """
        Fetch data for all tracked players using their IDs
        
        Args:
            source: Data source (fbref, sofascore, both)
        
        Returns:
            Summary of fetched data
        """
        tracked_players = self.db.query(Player).filter(Player.tracked == True).all()
        
        if not tracked_players:
            return {
                "status": "error",
                "message": "No tracked players found",
                "fetched_count": 0
            }
        
        fetched_count = 0
        
        for player in tracked_players:
            try:
                if source in ["fbref", "both"] and player.fbref_id:
                    # Fetch FBref data using player ID
                    fbref_data = self._fetch_fbref_by_id(player)
                    if fbref_data:
                        fetched_count += 1
                
                if source in ["sofascore", "both"] and player.sofascore_id:
                    # Fetch SofaScore data using player ID
                    sofa_data = self._fetch_sofascore_by_id(player)
                    if sofa_data:
                        fetched_count += 1
                        
            except Exception as e:
                self.errors.append(f"{player.name}: {str(e)}")
                continue
        
        self.db.commit()
        
        return {
            "status": "success",
            "tracked_players": len(tracked_players),
            "fetched_count": fetched_count,
            "errors": self.errors
        }
    
    def _fetch_fbref_by_id(self, player: Player) -> Optional[Dict[str, Any]]:
        """
        Fetch player data from FBref using player ID
        
        FBref player URL format: https://fbref.com/en/players/{player_id}/
        """
        try:
            # Construct FBref player URL
            player_url = f"https://fbref.com/en/players/{player.fbref_id}/"
            
            # This would need actual implementation with Selenium
            # For now, placeholder
            # In real implementation:
            # 1. Scrape player page
            # 2. Extract stats table
            # 3. Parse and save to database
            
            print(f"Would fetch FBref data for {player.name} from {player_url}")
            return {"success": True, "source": "fbref"}
            
        except Exception as e:
            self.errors.append(f"FBref fetch error for {player.name}: {str(e)}")
            return None
    
    def _fetch_sofascore_by_id(self, player: Player) -> Optional[Dict[str, Any]]:
        """
        Fetch player data from SofaScore using player ID
        
        SofaScore API: https://www.sofascore.com/api/v1/player/{player_id}/statistics
        """
        try:
            # Fetch player stats from SofaScore
            season = datetime.now().year
            stats_df = fetch_sofascore_player_stats(player.sofascore_id, str(season))
            
            if stats_df is not None and not stats_df.empty:
                # Save stats to database
                for idx, row in stats_df.iterrows():
                    stats = PlayerStats(
                        player_id=player.id,
                        date=datetime.now().date(),
                        minutes_played=int(row.get('minutesPlayed', 0) or 0),
                        goals=int(row.get('goals', 0) or 0),
                        assists=int(row.get('assists', 0) or 0),
                        advanced_stats=row.to_dict(),
                        source='sofascore'
                    )
                    self.db.add(stats)
                
                return {"success": True, "source": "sofascore"}
            
            return None
            
        except Exception as e:
            self.errors.append(f"SofaScore fetch error for {player.name}: {str(e)}")
            return None


def fetch_sofascore_player_stats(player_id: int, season: str = "2024") -> Optional[pd.DataFrame]:
    """
    Fetch player statistics from SofaScore using player ID
    
    Args:
        player_id: SofaScore player ID
        season: Season year (e.g., "2024")
    
    Returns:
        DataFrame with player statistics or None
    """
    try:
        url = f"https://www.sofascore.com/api/v1/player/{player_id}/statistics/season/{season}"
        
        if scrape_sofascore is None:
            print("SofaScore scraper not available")
            return None
        
        data = scrape_sofascore(url)
        
        if data and 'statistics' in data:
            stats_list = []
            
            for tournament_stats in data['statistics']:
                tournament = tournament_stats.get('tournament', {})
                stats = tournament_stats.get('statistics', {})
                
                stats_dict = {
                    'tournament': tournament.get('name'),
                    'tournament_id': tournament.get('id'),
                    'season': season
                }
                stats_dict.update(stats)
                stats_list.append(stats_dict)
            
            return pd.DataFrame(stats_list)
        
        return None
        
    except Exception as e:
        print(f"Error fetching SofaScore stats: {e}")
        return None
