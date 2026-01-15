"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime


# ===== Player Schemas =====

class PlayerBase(BaseModel):
    name: str
    position: str
    detailed_position: Optional[str] = None
    birth_date: Optional[date] = None
    nationality: Optional[str] = None
    tracked: bool = False
    fbref_id: Optional[str] = None
    sofascore_id: Optional[int] = None


class PlayerCreate(PlayerBase):
    current_team_id: Optional[int] = None


class PlayerUpdate(BaseModel):
    name: Optional[str] = None
    position: Optional[str] = None
    detailed_position: Optional[str] = None
    tracked: Optional[bool] = None
    current_team_id: Optional[int] = None


class TeamInfo(BaseModel):
    id: int
    name: str
    league: str
    
    class Config:
        from_attributes = True


class PlayerStatsSummary(BaseModel):
    minutes: int
    goals: int
    assists: int
    goals_per90: float
    assists_per90: float


class PlayerEvaluationSummary(BaseModel):
    overall_score: Optional[float] = None
    position_percentile: Optional[float] = None
    league_adjusted_score: Optional[float] = None
    form_trend: Optional[str] = None


class PlayerResponse(PlayerBase):
    id: int
    team: Optional[TeamInfo] = None
    current_season_stats: Optional[PlayerStatsSummary] = None
    latest_evaluation: Optional[PlayerEvaluationSummary] = None
    
    class Config:
        from_attributes = True


class PlayerListItem(BaseModel):
    id: int
    name: str
    position: str
    team: Optional[str] = None
    league: Optional[str] = None
    tracked: bool
    
    class Config:
        from_attributes = True


class PlayerListResponse(BaseModel):
    players: List[PlayerListItem]
    total: int


# ===== Stats Schemas =====

class PlayerStatsBase(BaseModel):
    player_id: int
    match_id: Optional[int] = None
    date: date
    minutes_played: int = 0
    goals: int = 0
    assists: int = 0
    shots: int = 0
    shots_on_target: int = 0
    pass_completion: Optional[float] = None
    passes_attempted: int = 0
    passes_completed: int = 0
    key_passes: int = 0
    progressive_passes: int = 0
    tackles: int = 0
    interceptions: int = 0
    blocks: int = 0
    clearances: int = 0
    dribbles_attempted: int = 0
    dribbles_completed: int = 0
    advanced_stats: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


class PlayerStatsCreate(PlayerStatsBase):
    pass


class PlayerStatsResponse(PlayerStatsBase):
    id: int
    opponent: Optional[str] = None
    
    class Config:
        from_attributes = True


class PlayerStatsListResponse(BaseModel):
    player_id: int
    period: str
    stats: List[PlayerStatsResponse]


# ===== Evaluation Schemas =====

class MetricBreakdown(BaseModel):
    attacking: float
    defensive: float
    possession: float
    efficiency: float
    consistency: float


class FormTrend(BaseModel):
    direction: str  # improving, declining, stable
    last_5_avg: float
    last_10_avg: float


class PeerComparison(BaseModel):
    position: str
    league: str
    better_than_pct: float


class PlayerEvaluationResponse(BaseModel):
    player_id: int
    evaluation_date: datetime
    period: str
    overall_score: float
    position_percentile: float
    league_adjusted_score: float
    breakdown: MetricBreakdown
    form_trend: FormTrend
    peer_comparison: PeerComparison
    
    class Config:
        from_attributes = True


# ===== League Schemas =====

class LeagueBase(BaseModel):
    name: str
    country: str
    tier: int = 1


class LeagueCreate(LeagueBase):
    pass


class LeagueStrengthInfo(BaseModel):
    date: date
    elo: float
    rank: Optional[int] = None


class LeagueResponse(LeagueBase):
    id: int
    current_elo: Optional[float] = None
    elo_rank: Optional[int] = None
    
    class Config:
        from_attributes = True


class LeagueListResponse(BaseModel):
    leagues: List[LeagueResponse]


class LeagueStrengthResponse(BaseModel):
    league_id: int
    name: str
    elo_history: List[LeagueStrengthInfo]


# ===== Team Schemas =====

class TeamBase(BaseModel):
    name: str
    league_id: int
    country: Optional[str] = None


class TeamCreate(TeamBase):
    pass


class TeamResponse(TeamBase):
    id: int
    league_name: Optional[str] = None
    
    class Config:
        from_attributes = True


# ===== Comparison Schemas =====

class PlayerComparisonRequest(BaseModel):
    player_ids: List[int] = Field(..., min_length=2, max_length=4)
    metrics: List[str]
    normalize_by_league: bool = False


class PlayerMetrics(BaseModel):
    id: int
    name: str
    metrics: Dict[str, float]
    league_adjusted: Optional[Dict[str, float]] = None


class PlayerComparisonResponse(BaseModel):
    players: List[PlayerMetrics]


# ===== Import Schemas =====

class ImportStatsRequest(BaseModel):
    source: str  # fbref, sofascore
    auto_match_players: bool = True


class ImportStatsResponse(BaseModel):
    status: str
    imported_records: int
    matched_players: int
    unmatched_players: int
    errors: List[str] = []


# ===== Trend Schemas =====

class TrendDataPoint(BaseModel):
    date: date
    value: float


class PlayerTrendResponse(BaseModel):
    player_id: int
    metric: str
    data_points: List[TrendDataPoint]
