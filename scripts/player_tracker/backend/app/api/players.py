"""
Player API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date, datetime, timedelta

from app.database import get_db
from app.models import Player, Team, League, PlayerStats, PlayerEvaluation
from app.schemas import (
    PlayerListResponse, PlayerListItem, PlayerResponse,
    PlayerCreate, PlayerUpdate, PlayerStatsListResponse,
    PlayerStatsResponse, TeamInfo, PlayerStatsSummary,
    PlayerEvaluationSummary
)
from app.utils.calculations import calculate_per90

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=PlayerListResponse)
def get_players(
    tracked: Optional[bool] = None,
    position: Optional[str] = None,
    league_id: Optional[int] = None,
    min_minutes: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get list of players with optional filters
    """
    query = db.query(Player)
    
    if tracked is not None:
        query = query.filter(Player.tracked == tracked)
    
    if position:
        query = query.filter(Player.position == position.upper())
    
    if league_id:
        query = query.join(Team).filter(Team.league_id == league_id)
    
    players = query.all()
    
    # Filter by minimum minutes if specified
    if min_minutes:
        filtered_players = []
        for player in players:
            total_minutes = db.query(PlayerStats).filter(
                PlayerStats.player_id == player.id
            ).with_entities(
                db.func.sum(PlayerStats.minutes_played)
            ).scalar() or 0
            
            if total_minutes >= min_minutes:
                filtered_players.append(player)
        players = filtered_players
    
    # Convert to response format
    player_items = []
    for player in players:
        player_items.append(PlayerListItem(
            id=player.id,
            name=player.name,
            position=player.position,
            team=player.team.name if player.team else None,
            league=player.team.league.name if player.team and player.team.league else None,
            tracked=player.tracked
        ))
    
    return PlayerListResponse(
        players=player_items,
        total=len(player_items)
    )


@router.get("/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, db: Session = Depends(get_db)):
    """
    Get detailed player information
    """
    player = db.query(Player).filter(Player.id == player_id).first()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get current season stats (last 365 days)
    one_year_ago = datetime.now().date() - timedelta(days=365)
    season_stats = db.query(PlayerStats).filter(
        PlayerStats.player_id == player_id,
        PlayerStats.date >= one_year_ago
    ).all()
    
    # Calculate aggregated stats
    total_minutes = sum(s.minutes_played for s in season_stats)
    total_goals = sum(s.goals for s in season_stats)
    total_assists = sum(s.assists for s in season_stats)
    
    current_season_stats = PlayerStatsSummary(
        minutes=total_minutes,
        goals=total_goals,
        assists=total_assists,
        goals_per90=calculate_per90(total_goals, total_minutes),
        assists_per90=calculate_per90(total_assists, total_minutes)
    )
    
    # Get latest evaluation
    latest_eval = db.query(PlayerEvaluation).filter(
        PlayerEvaluation.player_id == player_id
    ).order_by(PlayerEvaluation.evaluation_date.desc()).first()
    
    latest_evaluation = None
    if latest_eval:
        form_direction = "stable"
        if latest_eval.metric_breakdown:
            form_info = latest_eval.metric_breakdown.get("form_trend", {})
            form_direction = form_info.get("direction", "stable")
        
        latest_evaluation = PlayerEvaluationSummary(
            overall_score=latest_eval.overall_score,
            position_percentile=latest_eval.position_percentile,
            league_adjusted_score=latest_eval.league_adjusted_score,
            form_trend=form_direction
        )
    
    # Build team info
    team_info = None
    if player.team:
        team_info = TeamInfo(
            id=player.team.id,
            name=player.team.name,
            league=player.team.league.name if player.team.league else "Unknown"
        )
    
    return PlayerResponse(
        id=player.id,
        name=player.name,
        position=player.position,
        detailed_position=player.detailed_position,
        birth_date=player.birth_date,
        nationality=player.nationality,
        tracked=player.tracked,
        team=team_info,
        current_season_stats=current_season_stats if total_minutes > 0 else None,
        latest_evaluation=latest_evaluation
    )


@router.get("/{player_id}/stats", response_model=PlayerStatsListResponse)
def get_player_stats(
    player_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    per90: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get player statistics for a period
    """
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    query = db.query(PlayerStats).filter(PlayerStats.player_id == player_id)
    
    if start_date:
        query = query.filter(PlayerStats.date >= start_date)
    if end_date:
        query = query.filter(PlayerStats.date <= end_date)
    
    stats = query.order_by(PlayerStats.date.desc()).all()
    
    # Format period string
    if start_date and end_date:
        period = f"{start_date} to {end_date}"
    elif start_date:
        period = f"from {start_date}"
    elif end_date:
        period = f"until {end_date}"
    else:
        period = "all time"
    
    # Convert to response format
    stats_list = []
    for stat in stats:
        # Get opponent info if match exists
        opponent = None
        if stat.match:
            if stat.match.home_team_id == player.current_team_id:
                opponent = stat.match.away_team.name
            else:
                opponent = stat.match.home_team.name
        
        stats_list.append(PlayerStatsResponse(
            id=stat.id,
            player_id=stat.player_id,
            match_id=stat.match_id,
            date=stat.date,
            minutes_played=stat.minutes_played,
            goals=stat.goals,
            assists=stat.assists,
            shots=stat.shots,
            shots_on_target=stat.shots_on_target,
            pass_completion=stat.pass_completion,
            passes_attempted=stat.passes_attempted,
            passes_completed=stat.passes_completed,
            key_passes=stat.key_passes,
            progressive_passes=stat.progressive_passes,
            tackles=stat.tackles,
            interceptions=stat.interceptions,
            blocks=stat.blocks,
            clearances=stat.clearances,
            dribbles_attempted=stat.dribbles_attempted,
            dribbles_completed=stat.dribbles_completed,
            advanced_stats=stat.advanced_stats,
            source=stat.source,
            opponent=opponent
        ))
    
    return PlayerStatsListResponse(
        player_id=player_id,
        period=period,
        stats=stats_list
    )


@router.post("", response_model=PlayerResponse)
def create_player(player_data: PlayerCreate, db: Session = Depends(get_db)):
    """
    Create a new player
    """
    new_player = Player(
        name=player_data.name,
        position=player_data.position.upper(),
        detailed_position=player_data.detailed_position,
        birth_date=player_data.birth_date,
        nationality=player_data.nationality,
        current_team_id=player_data.current_team_id,
        tracked=player_data.tracked
    )
    
    db.add(new_player)
    db.commit()
    db.refresh(new_player)
    
    return get_player(new_player.id, db)


@router.patch("/{player_id}", response_model=PlayerResponse)
def update_player(
    player_id: int,
    player_data: PlayerUpdate,
    db: Session = Depends(get_db)
):
    """
    Update player information
    """
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    update_data = player_data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if field == "position" and value:
            value = value.upper()
        setattr(player, field, value)
    
    db.commit()
    db.refresh(player)
    
    return get_player(player_id, db)
