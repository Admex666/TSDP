"""
SQLAlchemy database models
"""
from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class League(Base):
    __tablename__ = "leagues"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    country = Column(String, nullable=False)
    tier = Column(Integer, default=1)
    
    # Relationships
    teams = relationship("Team", back_populates="league")
    league_strengths = relationship("LeagueStrength", back_populates="league")


class LeagueStrength(Base):
    __tablename__ = "league_strengths"
    
    id = Column(Integer, primary_key=True, index=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    elo_rating = Column(Float, nullable=False)
    rank = Column(Integer)
    
    # Relationships
    league = relationship("League", back_populates="league_strengths")


class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    country = Column(String)
    
    # Relationships
    league = relationship("League", back_populates="teams")
    players = relationship("Player", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    position = Column(String, nullable=True, index=True)  # GK, DEF, MID, ATT - can be null initially
    detailed_position = Column(String)  # CB, LB, CM, ST, etc.
    birth_date = Column(Date)
    nationality = Column(String)
    current_team_id = Column(Integer, ForeignKey("teams.id"))
    tracked = Column(Boolean, default=False, index=True)
    
    # External IDs for scraping
    fbref_id = Column(String, unique=True, index=True)  # FBref player ID
    sofascore_id = Column(Integer, unique=True, index=True)  # SofaScore player ID
    
    # Relationships
    team = relationship("Team", back_populates="players")
    stats = relationship("PlayerStats", back_populates="player")
    evaluations = relationship("PlayerEvaluation", back_populates="player")


class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    finished = Column(Boolean, default=False)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    player_stats = relationship("PlayerStats", back_populates="match")


class PlayerStats(Base):
    __tablename__ = "player_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Basic stats
    minutes_played = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    
    # Passing
    pass_completion = Column(Float)  # percentage
    passes_attempted = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)
    progressive_passes = Column(Integer, default=0)
    
    # Defensive
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    
    # Dribbling
    dribbles_attempted = Column(Integer, default=0)
    dribbles_completed = Column(Integer, default=0)
    
    # Advanced stats (stored as JSON for flexibility)
    advanced_stats = Column(JSON)
    
    # Source
    source = Column(String)  # fbref, sofascore, etc.
    
    # Relationships
    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="player_stats")


class PlayerEvaluation(Base):
    __tablename__ = "player_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, index=True)
    evaluation_date = Column(DateTime, default=datetime.utcnow, index=True)
    evaluation_period = Column(String)  # season, last_10_games, last_30_days
    
    # Overall scores
    overall_score = Column(Float)
    position_percentile = Column(Float)  # 0-100
    league_adjusted_score = Column(Float)
    form_trend = Column(Float)
    
    # Minutes played in evaluation period
    minutes_played = Column(Integer)
    
    # Breakdown by dimension (stored as JSON)
    metric_breakdown = Column(JSON)
    
    # Evaluation version
    evaluation_version = Column(String, default="standard")
    
    # Relationships
    player = relationship("Player", back_populates="evaluations")
