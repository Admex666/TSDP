"""
API endpoints for enhanced scraper imports
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.schemas import ImportStatsResponse
from app.services.enhanced_import import EnhancedImportService

router = APIRouter(prefix="/scraper", tags=["scraper"])


@router.post("/fbref/league", response_model=ImportStatsResponse)
def import_fbref_league(
    countrycode: str = Query(..., description="League code (e.g., HUN, GER, ENG)"),
    season: Optional[str] = Query(None, description="Season (e.g., 2024-2025)"),
    auto_create_players: bool = Query(True, description="Auto-create players if not found"),
    db: Session = Depends(get_db)
):
    """
    Import entire league player data from FBref
    
    Examples:
    - HUN (NB I)
    - GER (Bundesliga)
    - ENG (Premier League)
    - Big5 (Top 5 leagues combined)
    """
    import_service = EnhancedImportService(db)
    result = import_service.import_fbref_league(countrycode, season, auto_create_players)
    
    return ImportStatsResponse(**result)


@router.post("/sofascore/match", response_model=ImportStatsResponse)
def import_sofascore_match(
    event_id: int = Query(..., description="SofaScore event/match ID"),
    auto_create_players: bool = Query(True, description="Auto-create players if not found"),
    db: Session = Depends(get_db)
):
    """
    Import match data from SofaScore
    
    Requires SofaScore event ID (found in match URL)
    """
    import_service = EnhancedImportService(db)
    result = import_service.import_sofascore_match(event_id, auto_create_players)
    
    return ImportStatsResponse(**result)
