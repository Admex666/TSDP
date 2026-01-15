"""
League API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import date

from app.database import get_db
from app.models import League, LeagueStrength
from app.schemas import (
    LeagueListResponse, LeagueResponse,
    LeagueStrengthResponse, LeagueStrengthInfo
)

router = APIRouter(prefix="/leagues", tags=["leagues"])


@router.get("", response_model=LeagueListResponse)
def get_leagues(db: Session = Depends(get_db)):
    """
    Get list of all leagues
    """
    leagues = db.query(League).all()
    
    league_responses = []
    for league in leagues:
        # Get latest strength data
        latest_strength = db.query(LeagueStrength).filter(
            LeagueStrength.league_id == league.id
        ).order_by(LeagueStrength.date.desc()).first()
        
        league_responses.append(LeagueResponse(
            id=league.id,
            name=league.name,
            country=league.country,
            tier=league.tier,
            current_elo=latest_strength.elo_rating if latest_strength else None,
            elo_rank=latest_strength.rank if latest_strength else None
        ))
    
    return LeagueListResponse(leagues=league_responses)


@router.get("/{league_id}/strength", response_model=LeagueStrengthResponse)
def get_league_strength(
    league_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db)
):
    """
    Get league strength history (Elo ratings)
    """
    league = db.query(League).filter(League.id == league_id).first()
    if not league:
        raise HTTPException(status_code=404, detail="League not found")
    
    query = db.query(LeagueStrength).filter(LeagueStrength.league_id == league_id)
    
    if start_date:
        query = query.filter(LeagueStrength.date >= start_date)
    if end_date:
        query = query.filter(LeagueStrength.date <= end_date)
    
    strengths = query.order_by(LeagueStrength.date).all()
    
    elo_history = [
        LeagueStrengthInfo(
            date=s.date,
            elo=s.elo_rating,
            rank=s.rank
        )
        for s in strengths
    ]
    
    return LeagueStrengthResponse(
        league_id=league_id,
        name=league.name,
        elo_history=elo_history
    )
