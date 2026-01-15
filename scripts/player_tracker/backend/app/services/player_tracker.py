"""
Player-specific scraper service
Allows tracking specific players and fetching their data from FBref and SofaScore
"""
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from sqlalchemy.orm import Session
import re

from app.models import Player, Team, League, PlayerStats, Match
from app.services import fbref_scraper, sofascore_scraper
from app.utils.matching import match_player, extract_position_group
from config import config


class PlayerTrackerService:
    """
    Service for tracking specific players and fetching their data
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.errors = []
    
    def add_tracked_players(
        self,
        player_names: List[str],
        auto_search: bool = True
    ) -> Dict[str, Any]:
        """
        Add players to tracking list
        
        Args:
            player_names: List of player names to track
            auto_search: Automatically search for players on FBref/SofaScore
        
        Returns:
            Summary dict with added/found players
        """
        added = []
        already_exists = []
        not_found = []
        
        for name in player_names:
            # Check if player already exists
            existing = self.db.query(Player).filter(
                Player.name.ilike(f"%{name}%")
            ).first()
            
            if existing:
                if not existing.tracked:
                    existing.tracked = True
                    added.append(existing.name)
                else:
                    already_exists.append(existing.name)
            elif auto_search:
                # Try to find player on FBref
                player_data = self._search_player_fbref(name)
                if player_data:
                    new_player = self._create_player_from_search(player_data)
                    added.append(new_player.name)
                else:
                    not_found.append(name)
            else:
                not_found.append(name)
        
        self.db.commit()
        
        return {
            "status": "success",
            "added": added,
            "already_tracked": already_exists,
            "not_found": not_found
        }
    
    def fetch_tracked_players_data(
        self,
        source: str = "both",  # fbref, sofascore, both
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch data for all tracked players
        
        Args:
            source: Data source (fbref, sofascore, both)
            season: Season for FBref data
        
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
                if source in ["fbref", "both"]:
                    # Fetch FBref data
                    fbref_data = self._fetch_player_fbref(player, season)
                    if fbref_data:
                        fetched_count += 1
                
                if source in ["sofascore", "both"]:
                    # Fetch SofaScore data
                    sofa_data = self._fetch_player_sofascore(player)
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
    
    def _search_player_fbref(self, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a player on FBref
        Uses FBref search functionality
        """
        try:
            # FBref search URL
            search_url = f"https://fbref.com/en/search/search.fcgi?search={player_name.replace(' ', '+')}"
            
            # This would need actual implementation with Selenium
            # For now, return None (placeholder)
            # In real implementation, parse search results and return player data
            
            return None
            
        except Exception as e:
            self.errors.append(f"FBref search error for {player_name}: {str(e)}")
            return None
    
    def _fetch_player_fbref(
        self,
        player: Player,
        season: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch specific player data from FBref
        """
        try:
            # If player has team and league, fetch from league data
            if player.team and player.team.league:
                league_name = player.team.league.name
                
                # Map league name to country code
                country_map = {
                    "NB I": "HUN",
                    "Bundesliga": "GER",
                    "Premier League": "ENG",
                    "La Liga": "ESP",
                    "Serie A": "ITA",
                    "Ligue 1": "FRA",
                    "Eredivisie": "NED"
                }
                
                countrycode = country_map.get(league_name)
                if countrycode:
                    # Fetch league data and filter for this player
                    df = fbref_scraper.get_all_player_data(countrycode, season)
                    player_data = df[df['Player'].str.contains(player.name, case=False, na=False)]
                    
                    if not player_data.empty:
                        # Create stats from player data
                        row = player_data.iloc[0]
                        stats = self._create_fbref_stats(player.id, row, season)
                        self.db.add(stats)
                        return {"success": True, "source": "fbref"}
            
            return None
            
        except Exception as e:
            self.errors.append(f"FBref fetch error for {player.name}: {str(e)}")
            return None
    
    def _fetch_player_sofascore(self, player: Player) -> Optional[Dict[str, Any]]:
        """
        Fetch specific player data from SofaScore
        """
        try:
            # This would need SofaScore player ID
            # For now, placeholder
            # In real implementation, search for player and get their recent matches
            
            return None
            
        except Exception as e:
            self.errors.append(f"SofaScore fetch error for {player.name}: {str(e)}")
            return None
    
    def _create_player_from_search(self, player_data: Dict[str, Any]) -> Player:
        """
        Create player from search results
        """
        # Extract data from search results
        name = player_data.get('name', '')
        position = player_data.get('position', 'MF')
        team_name = player_data.get('team', '')
        
        # Get or create team
        team = None
        if team_name:
            team = self.db.query(Team).filter(Team.name.ilike(f"%{team_name}%")).first()
        
        # Create player
        position_group = extract_position_group(position)
        new_player = Player(
            name=name,
            position=position_group,
            detailed_position=position,
            current_team_id=team.id if team else None,
            tracked=True
        )
        
        self.db.add(new_player)
        self.db.flush()
        
        return new_player
    
    def _create_fbref_stats(
        self,
        player_id: int,
        row: pd.Series,
        season: Optional[str]
    ) -> PlayerStats:
        """Create PlayerStats from FBref row"""
        if season:
            year = int(season.split('-')[0])
            stats_date = date(year, 8, 1)
        else:
            stats_date = datetime.now().date()
        
        return PlayerStats(
            player_id=player_id,
            date=stats_date,
            minutes_played=int(row.get('Playing Time_Min', 0) or 0),
            goals=int(row.get('Performance_Gls', 0) or 0),
            assists=int(row.get('Performance_Ast', 0) or 0),
            shots=int(row.get('Performance_Sh', 0) or 0),
            shots_on_target=int(row.get('Performance_SoT', 0) or 0),
            pass_completion=float(row.get('Total_Cmp%', 0) or 0) if 'Total_Cmp%' in row else None,
            passes_attempted=int(row.get('Total_Att', 0) or 0) if 'Total_Att' in row else 0,
            passes_completed=int(row.get('Total_Cmp', 0) or 0) if 'Total_Cmp' in row else 0,
            key_passes=int(row.get('KP', 0) or 0) if 'KP' in row else 0,
            progressive_passes=int(row.get('PrgP', 0) or 0) if 'PrgP' in row else 0,
            tackles=int(row.get('Tackles_Tkl', 0) or 0) if 'Tackles_Tkl' in row else 0,
            interceptions=int(row.get('Int', 0) or 0) if 'Int' in row else 0,
            blocks=int(row.get('Blocks_Blocks', 0) or 0) if 'Blocks_Blocks' in row else 0,
            clearances=int(row.get('Clr', 0) or 0) if 'Clr' in row else 0,
            dribbles_attempted=int(row.get('Take-Ons_Att', 0) or 0) if 'Take-Ons_Att' in row else 0,
            dribbles_completed=int(row.get('Take-Ons_Succ', 0) or 0) if 'Take-Ons_Succ' in row else 0,
            source='fbref'
        )


def get_sofascore_player_id(player_name: str, team_name: Optional[str] = None) -> Optional[int]:
    """
    Search for player on SofaScore and get their ID
    
    Args:
        player_name: Player name to search
        team_name: Optional team name for better matching
    
    Returns:
        SofaScore player ID or None
    """
    try:
        import tls_client
        
        sess = tls_client.Session(client_identifier="chrome_118")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.sofascore.com/",
        }
        
        # SofaScore search API
        search_url = f"https://www.sofascore.com/api/v1/search/all?q={player_name.replace(' ', '%20')}"
        
        resp = sess.get(search_url, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Look for player in results
            if 'results' in data:
                for result in data['results']:
                    if result.get('type') == 'player':
                        player_data = result.get('entity', {})
                        
                        # If team name provided, try to match
                        if team_name:
                            player_team = player_data.get('team', {}).get('name', '')
                            if team_name.lower() in player_team.lower():
                                return player_data.get('id')
                        else:
                            # Return first player match
                            return player_data.get('id')
        
        return None
        
    except Exception as e:
        print(f"SofaScore search error: {e}")
        return None


def fetch_sofascore_player_stats(player_id: int, season: str = "2024") -> Optional[pd.DataFrame]:
    """
    Fetch player statistics from SofaScore
    
    Args:
        player_id: SofaScore player ID
        season: Season year (e.g., "2024")
    
    Returns:
        DataFrame with player statistics or None
    """
    try:
        url = f"https://www.sofascore.com/api/v1/player/{player_id}/statistics/season/{season}"
        
        data = sofascore_scraper.scrape_sofascore(url)
        
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
