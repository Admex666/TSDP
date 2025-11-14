"""
SQLAlchemy models for CS:GO scraper database
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class ScrapeLog(Base):
    __tablename__ = 'scrape_log'
    
    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False, index=True)
    scrape_type = Column(String, nullable=False)  # 'match', 'h2h', 'team_history', 'rankings'
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_msg = Column(Text, nullable=True)
    event_id = Column(String, nullable=True)
    match_id = Column(String, nullable=True)
    team_id = Column(String, nullable=True)


class Event(Base):
    __tablename__ = 'events'
    
    event_id = Column(String, primary_key=True)
    event_name = Column(String, nullable=False)
    url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    matches = relationship("Match", back_populates="event")


class Match(Base):
    __tablename__ = 'matches'
    
    match_id = Column(String, primary_key=True)
    event_id = Column(String, ForeignKey('events.event_id'))
    date = Column(DateTime, nullable=False)
    date_month = Column(Integer)
    date_day = Column(Integer)
    team_home = Column(String, nullable=False)
    team_home_id = Column(String, nullable=True)
    team_away = Column(String, nullable=False)
    team_away_id = Column(String, nullable=True)
    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    rounds = Column(Integer, nullable=True)
    url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    event = relationship("Event", back_populates="matches")
    h2h = relationship("MatchH2H", back_populates="match", uselist=False)


class MatchH2H(Base):
    __tablename__ = 'match_h2h'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String, ForeignKey('matches.match_id'), unique=True)
    home_team = Column(String)
    home_team_id = Column(String)
    away_team = Column(String)
    away_team_id = Column(String)
    
    # H2H stats
    wins_home = Column(Integer)
    wins_away = Column(Integer)
    total_non_overtime = Column(Integer)
    home_win_rate = Column(Float)
    
    # Team stats
    home_team_avg_rating = Column(Float)
    home_team_std_rating = Column(Float)
    home_team_avg_ADR = Column(Float)
    home_team_std_ADR = Column(Float)
    home_team_avg_Swing = Column(Float)
    home_team_std_Swing = Column(Float)
    
    away_team_avg_rating = Column(Float)
    away_team_std_rating = Column(Float)
    away_team_avg_ADR = Column(Float)
    away_team_std_ADR = Column(Float)
    away_team_avg_Swing = Column(Float)
    away_team_std_Swing = Column(Float)
    
    # Map stats
    maps_played = Column(Integer)
    home_maps_won = Column(Integer)
    away_maps_won = Column(Integer)
    home_maps_picked = Column(Integer)
    map_avg_score_diff = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    match = relationship("Match", back_populates="h2h")


class TeamHistory(Base):
    __tablename__ = 'team_history'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(String, nullable=False, index=True)
    match_id = Column(String, nullable=False)
    match_date = Column(DateTime, nullable=False)
    opponent_name = Column(String, nullable=False)
    result = Column(String, nullable=False)  # 'win' or 'loss'
    score_for = Column(Integer)
    score_against = Column(Integer)
    map_type = Column(String)
    url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Ranking(Base):
    __tablename__ = 'rankings'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    team_id = Column(String, nullable=False, index=True)
    team_name = Column(String, nullable=False, index=True)
    points = Column(Integer, nullable=False)
    profile_link = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Odds(Base):
    __tablename__ = 'odds'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    home_team_mapped = Column(String, nullable=True)
    away_team_mapped = Column(String, nullable=True)
    home_odds = Column(Float)
    away_odds = Column(Float)
    home_implied_odds = Column(Float)
    away_implied_odds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class MLFeature(Base):
    __tablename__ = 'ml_features'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(String, nullable=False, index=True)
    
    # Rolling features
    home_last_3_winrate = Column(Float)
    home_last_5_winrate = Column(Float)
    home_last_3_avg_score_for = Column(Float)
    home_last_3_avg_score_against = Column(Float)
    home_current_streak = Column(Integer)
    
    away_last_3_winrate = Column(Float)
    away_last_5_winrate = Column(Float)
    away_last_3_avg_score_for = Column(Float)
    away_last_3_avg_score_against = Column(Float)
    away_current_streak = Column(Integer)
    
    diff_last_3_winrate = Column(Float)
    
    # Historical H2H averages
    home_avg_h2h_winrate = Column(Float)
    home_avg_rating = Column(Float)
    home_avg_rating_std = Column(Float)
    home_avg_adr = Column(Float)
    home_avg_adr_std = Column(Float)
    home_avg_swing = Column(Float)
    home_avg_swing_std = Column(Float)
    home_avg_maps_total = Column(Float)
    home_avg_map_winrate = Column(Float)
    home_avg_map_pickrate = Column(Float)
    home_avg_map_score_diff = Column(Float)
    home_avg_rank_diff = Column(Float)
    home_std_rank_diff = Column(Float)
    home_avg_point_diff = Column(Float)
    home_std_point_diff = Column(Float)
    
    away_avg_h2h_winrate = Column(Float)
    away_avg_rating = Column(Float)
    away_avg_rating_std = Column(Float)
    away_avg_adr = Column(Float)
    away_avg_adr_std = Column(Float)
    away_avg_swing = Column(Float)
    away_avg_swing_std = Column(Float)
    away_avg_maps_total = Column(Float)
    away_avg_map_winrate = Column(Float)
    away_avg_map_pickrate = Column(Float)
    away_avg_map_score_diff = Column(Float)
    away_avg_rank_diff = Column(Float)
    away_std_rank_diff = Column(Float)
    away_avg_point_diff = Column(Float)
    away_std_point_diff = Column(Float)
    
    # Rankings
    home_current_rank = Column(Integer)
    away_current_rank = Column(Integer)
    home_rank_change = Column(Integer)
    away_rank_change = Column(Integer)
    
    # H2H between teams
    H2H_winrate_team1 = Column(Float)
    H2H_games = Column(Integer)
    
    # Odds
    home_odds = Column(Float)
    away_odds = Column(Float)
    home_implied_odds = Column(Float)
    away_implied_odds = Column(Float)
    
    # Label
    label_home_win = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)