"""
Enhanced import service with FBref and SofaScore integration
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session

from app.models import Player, Team, League, PlayerStats, Match
from app.utils.matching import match_player, match_team, extract_position_group
from app.services import fbref_scraper, sofascore_scraper
from config import config


class EnhancedImportService:
    """
    Enhanced service for importing player statistics from various sources
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.errors = []
        self.matched_players = set()
        self.unmatched_players = []
    
    def import_fbref_league(
        self,
        countrycode: str,
        season: Optional[str] = None,
        auto_create_players: bool = True
    ) -> Dict[str, Any]:
        """
        Import entire league data from FBref
        
        Args:
            countrycode: League code (e.g., 'HUN', 'GER', 'ENG')
            season: Season string (e.g., '2024-2025'), None for current
            auto_create_players: Create new players if not found
        
        Returns:
            Import summary dict
        """
        try:
            print(f"Fetching FBref data for {countrycode}...")
            df = fbref_scraper.get_all_player_data(countrycode, season)
            
            # Get or create league
            comp_id, league_name = fbref_scraper.team_dict_get(countrycode)
            league = self._get_or_create_league(league_name, countrycode)
            
            imported_count = 0
            db_players = self.db.query(Player).all()
            db_teams = self.db.query(Team).all()
            
            for idx, row in df.iterrows():
                try:
                    player_name = row.get('Player', '')
                    team_name = row.get('Squad', '')
                    position = row.get('Pos', 'MF')
                    
                    if not player_name:
                        continue
                    
                    # Match or create team
                    team = self._get_or_create_team(team_name, league.id, db_teams)
                    
                    # Match or create player
                    matched_player, score = match_player(player_name, team_name, db_players)
                    
                    if not matched_player and auto_create_players:
                        matched_player = self._create_player(
                            player_name, position, team.id, db_players
                        )
                    
                    if not matched_player:
                        self.unmatched_players.append(player_name)
                        continue
                    
                    self.matched_players.add(matched_player.id)
                    
                    # Create aggregated stats (season total)
                    stats = self._create_fbref_stats(matched_player.id, row, season)
                    self.db.add(stats)
                    imported_count += 1
                    
                except Exception as e:
                    self.errors.append(f"Row {idx}: {str(e)}")
                    continue
            
            self.db.commit()
            
            return {
                "status": "success",
                "imported_records": imported_count,
                "matched_players": len(self.matched_players),
                "unmatched_players": len(self.unmatched_players),
                "errors": self.errors
            }
            
        except Exception as e:
            self.db.rollback()
            return {
                "status": "error",
                "imported_records": 0,
                "matched_players": 0,
                "unmatched_players": 0,
                "errors": [f"FBref import failed: {str(e)}"]
            }
    
    def import_sofascore_match(
        self,
        event_id: int,
        auto_create_players: bool = True
    ) -> Dict[str, Any]:
        """
        Import match data from SofaScore
        
        Args:
            event_id: SofaScore event/match ID
            auto_create_players: Create new players if not found
        
        Returns:
            Import summary dict
        """
        try:
            print(f"Fetching SofaScore data for event {event_id}...")
            
            # Get lineups with statistics
            lineups_df = sofascore_scraper.create_lineups_df(event_id)
            
            imported_count = 0
            db_players = self.db.query(Player).all()
            db_teams = self.db.query(Team).all()
            
            # Create match record
            match = self._create_match_from_sofascore(event_id)
            
            for idx, row in lineups_df.iterrows():
                try:
                    player_name = row.get('name', '')
                    position = row.get('position', 'M')
                    team_side = row.get('team', 'home')
                    
                    if not player_name:
                        continue
                    
                    # Determine team
                    team_id = match.home_team_id if team_side == 'home' else match.away_team_id
                    team = self.db.query(Team).filter(Team.id == team_id).first()
                    
                    # Match or create player
                    matched_player, score = match_player(player_name, team.name if team else '', db_players)
                    
                    if not matched_player and auto_create_players:
                        matched_player = self._create_player(
                            player_name, position, team_id, db_players
                        )
                    
                    if not matched_player:
                        self.unmatched_players.append(player_name)
                        continue
                    
                    self.matched_players.add(matched_player.id)
                    
                    # Create match stats
                    stats = self._create_sofascore_stats(
                        matched_player.id, match.id, row
                    )
                    self.db.add(stats)
                    imported_count += 1
                    
                except Exception as e:
                    self.errors.append(f"Row {idx}: {str(e)}")
                    continue
            
            self.db.commit()
            
            return {
                "status": "success",
                "imported_records": imported_count,
                "matched_players": len(self.matched_players),
                "unmatched_players": len(self.unmatched_players),
                "errors": self.errors
            }
            
        except Exception as e:
            self.db.rollback()
            return {
                "status": "error",
                "imported_records": 0,
                "matched_players": 0,
                "unmatched_players": 0,
                "errors": [f"SofaScore import failed: {str(e)}"]
            }
    
    def _get_or_create_league(self, name: str, country: str) -> League:
        """Get existing league or create new one"""
        league = self.db.query(League).filter(League.name == name).first()
        if not league:
            league = League(name=name, country=country, tier=1)
            self.db.add(league)
            self.db.flush()
        return league
    
    def _get_or_create_team(self, name: str, league_id: int, db_teams: List[Team]) -> Team:
        """Get existing team or create new one"""
        team, _ = match_team(name, db_teams, threshold=75)
        if not team:
            team = Team(name=name, league_id=league_id)
            self.db.add(team)
            self.db.flush()
            db_teams.append(team)
        return team
    
    def _create_player(
        self,
        name: str,
        position: str,
        team_id: Optional[int],
        db_players: List[Player]
    ) -> Player:
        """Create a new player"""
        position_group = extract_position_group(position)
        
        new_player = Player(
            name=name,
            position=position_group,
            detailed_position=position,
            current_team_id=team_id,
            tracked=False
        )
        
        self.db.add(new_player)
        self.db.flush()
        db_players.append(new_player)
        
        return new_player
    
    def _create_fbref_stats(
        self,
        player_id: int,
        row: pd.Series,
        season: Optional[str]
    ) -> PlayerStats:
        """Create PlayerStats from FBref row"""
        # Determine date (use current date or season start)
        if season:
            year = int(season.split('-')[0])
            stats_date = date(year, 8, 1)  # Assume season starts in August
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
    
    def _create_sofascore_stats(
        self,
        player_id: int,
        match_id: int,
        row: pd.Series
    ) -> PlayerStats:
        """Create PlayerStats from SofaScore row"""
        return PlayerStats(
            player_id=player_id,
            match_id=match_id,
            date=datetime.now().date(),
            minutes_played=int(row.get('minutesPlayed', 0) or 0),
            goals=int(row.get('goals', 0) or 0),
            assists=int(row.get('assists', 0) or 0),
            shots=int(row.get('totalShots', 0) or 0),
            shots_on_target=int(row.get('shotsOnTarget', 0) or 0),
            pass_completion=float(row.get('accuratePass', 0) or 0) if 'accuratePass' in row else None,
            tackles=int(row.get('totalTackles', 0) or 0) if 'totalTackles' in row else 0,
            interceptions=int(row.get('interceptions', 0) or 0) if 'interceptions' in row else 0,
            dribbles_attempted=int(row.get('totalDuels', 0) or 0) if 'totalDuels' in row else 0,
            dribbles_completed=int(row.get('wonDuels', 0) or 0) if 'wonDuels' in row else 0,
            advanced_stats=row.to_dict(),  # Store all stats as JSON
            source='sofascore'
        )
    
    def _create_match_from_sofascore(self, event_id: int) -> Match:
        """Create Match record from SofaScore event (placeholder)"""
        # This would need actual match data from SofaScore API
        # For now, create a placeholder
        match = Match(
            date=datetime.now().date(),
            home_team_id=1,  # Placeholder
            away_team_id=2,  # Placeholder
            finished=True
        )
        self.db.add(match)
        self.db.flush()
        return match
