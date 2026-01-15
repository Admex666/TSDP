"""
Data import service for CSV/JSON files
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from app.models import Player, Team, League, PlayerStats
from app.utils.matching import match_player, match_team, extract_position_group
from config import config


class ImportService:
    """
    Service for importing player statistics from various sources
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.errors = []
        self.matched_players = set()
        self.unmatched_players = []
    
    def import_csv(
        self,
        file_path: str,
        source: str,
        auto_match: bool = True
    ) -> Dict[str, Any]:
        """
        Import player stats from CSV file
        
        Args:
            file_path: Path to CSV file
            source: Data source (fbref, sofascore)
            auto_match: Automatically match players to database
        
        Returns:
            Import summary dict
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return {
                "status": "error",
                "imported_records": 0,
                "matched_players": 0,
                "unmatched_players": 0,
                "errors": [f"Failed to read CSV: {str(e)}"]
            }
        
        if source.lower() == "fbref":
            return self._import_fbref_csv(df, auto_match)
        elif source.lower() == "sofascore":
            return self._import_sofascore_csv(df, auto_match)
        else:
            return {
                "status": "error",
                "imported_records": 0,
                "matched_players": 0,
                "unmatched_players": 0,
                "errors": [f"Unknown source: {source}"]
            }
    
    def _import_fbref_csv(self, df: pd.DataFrame, auto_match: bool) -> Dict[str, Any]:
        """
        Import FBref format CSV
        
        Expected columns: Player, Nation, Pos, Age, MP, Starts, Min, Gls, Ast, etc.
        """
        imported_count = 0
        
        # Get all players and teams from database for matching
        db_players = self.db.query(Player).all()
        db_teams = self.db.query(Team).all()
        
        for idx, row in df.iterrows():
            try:
                player_name = row.get("Player", "")
                position = row.get("Pos", "MF")
                team_name = row.get("Squad", "") or row.get("Team", "")
                
                if not player_name:
                    self.errors.append(f"Row {idx}: Missing player name")
                    continue
                
                # Match player
                matched_player = None
                if auto_match:
                    matched_player, score = match_player(
                        player_name,
                        team_name,
                        db_players
                    )
                    
                    if matched_player:
                        self.matched_players.add(matched_player.id)
                    else:
                        self.unmatched_players.append(player_name)
                        # Optionally create new player
                        if config.AUTO_CREATE_PLAYERS:
                            matched_player = self._create_player_from_row(
                                player_name, position, team_name, db_teams
                            )
                
                if not matched_player:
                    continue
                
                # Create stats record
                stats = PlayerStats(
                    player_id=matched_player.id,
                    date=datetime.now().date(),  # Use import date as default
                    minutes_played=int(row.get("Min", 0) or 0),
                    goals=int(row.get("Gls", 0) or 0),
                    assists=int(row.get("Ast", 0) or 0),
                    shots=int(row.get("Sh", 0) or 0),
                    shots_on_target=int(row.get("SoT", 0) or 0),
                    pass_completion=float(row.get("Cmp%", 0) or 0) if "Cmp%" in row else None,
                    passes_attempted=int(row.get("Att", 0) or 0) if "Att" in row else 0,
                    passes_completed=int(row.get("Cmp", 0) or 0) if "Cmp" in row else 0,
                    tackles=int(row.get("Tkl", 0) or 0) if "Tkl" in row else 0,
                    interceptions=int(row.get("Int", 0) or 0) if "Int" in row else 0,
                    blocks=int(row.get("Blocks", 0) or 0) if "Blocks" in row else 0,
                    source="fbref"
                )
                
                self.db.add(stats)
                imported_count += 1
                
            except Exception as e:
                self.errors.append(f"Row {idx}: {str(e)}")
                continue
        
        # Commit all changes
        try:
            self.db.commit()
            status = "success"
        except Exception as e:
            self.db.rollback()
            status = "error"
            self.errors.append(f"Database commit failed: {str(e)}")
        
        return {
            "status": status,
            "imported_records": imported_count,
            "matched_players": len(self.matched_players),
            "unmatched_players": len(self.unmatched_players),
            "errors": self.errors
        }
    
    def _import_sofascore_csv(self, df: pd.DataFrame, auto_match: bool) -> Dict[str, Any]:
        """
        Import SofaScore format CSV
        
        Expected columns may vary - implement based on actual SofaScore export format
        """
        # Placeholder - implement based on actual SofaScore format
        return {
            "status": "error",
            "imported_records": 0,
            "matched_players": 0,
            "unmatched_players": 0,
            "errors": ["SofaScore import not yet implemented"]
        }
    
    def _create_player_from_row(
        self,
        name: str,
        position: str,
        team_name: str,
        db_teams: List[Team]
    ) -> Player:
        """
        Create a new player from import data
        """
        # Extract position group
        position_group = extract_position_group(position)
        
        # Try to match team
        matched_team = None
        if team_name:
            matched_team, _ = match_team(team_name, db_teams)
        
        new_player = Player(
            name=name,
            position=position_group,
            detailed_position=position,
            current_team_id=matched_team.id if matched_team else None,
            tracked=False
        )
        
        self.db.add(new_player)
        self.db.flush()  # Get ID without committing
        
        return new_player
