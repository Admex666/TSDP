"""
Player tracking API endpoints - ID-based
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from app.database import get_db
from app.services.id_tracker import IDBasedTrackerService

router = APIRouter(prefix="/tracker", tags=["tracker"])


class PlayerIDInput(BaseModel):
    fbref_id: Optional[str] = None
    sofascore_id: Optional[int] = None
    name: Optional[str] = None  # For display purposes


class AddPlayersByIDRequest(BaseModel):
    players: List[PlayerIDInput]


class FetchDataRequest(BaseModel):
    source: str = "both"  # fbref, sofascore, both


@router.post("/add-players-by-id")
def add_players_by_id(
    request: AddPlayersByIDRequest,
    db: Session = Depends(get_db)
):
    """
    Add specific players to tracking list using their FBref and/or SofaScore IDs
    
    Example:
    ```json
    {
        "players": [
            {
                "fbref_id": "abc123",
                "sofascore_id": 12345,
                "name": "Szabó Dominik"
            },
            {
                "fbref_id": "def456",
                "name": "Nagy Ádám"
            },
            {
                "sofascore_id": 67890,
                "name": "Varga Barnabás"
            }
        ]
    }
    ```
    """
    service = IDBasedTrackerService(db)
    result = service.add_players_by_id(
        [p.dict() for p in request.players]
    )
    
    return result


@router.post("/fetch-data")
def fetch_tracked_players_data(
    request: FetchDataRequest,
    db: Session = Depends(get_db)
):
    """
    Fetch data for all tracked players from FBref and/or SofaScore using their IDs
    
    Example:
    ```json
    {
        "source": "both"
    }
    ```
    """
    service = IDBasedTrackerService(db)
    result = service.fetch_tracked_players_data(request.source)
    
    return result


@router.get("/tracked-players")
def get_tracked_players(db: Session = Depends(get_db)):
    """
    Get list of all tracked players with their IDs
    """
    from app.models import Player
    
    players = db.query(Player).filter(Player.tracked == True).all()
    
    return {
        "tracked_players": [
            {
                "id": p.id,
                "name": p.name,
                "position": p.position,
                "team": p.team.name if p.team else None,
                "league": p.team.league.name if p.team and p.team.league else None,
                "fbref_id": p.fbref_id,
                "sofascore_id": p.sofascore_id
            }
            for p in players
        ],
        "total": len(players)
    }
