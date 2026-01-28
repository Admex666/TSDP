import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Tournament(Base):
    __tablename__ = 'tournaments'
    id = Column(Integer, primary_key=True)  # SofaScore ID
    name = Column(String)
    slug = Column(String)
    matches = relationship("Match", back_populates="tournament")

class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)  # SofaScore ID
    name = Column(String)
    short_name = Column(String)

class Player(Base):
    __tablename__ = 'players'
    id = Column(Integer, primary_key=True)  # SofaScore ID
    name = Column(String)
    slug = Column(String)
    position = Column(String) # F, M, D, G
    
    match_stats = relationship("PlayerMatchStats", back_populates="player")

class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)  # SofaScore Event ID
    tournament_id = Column(Integer, ForeignKey('tournaments.id'))
    season_id = Column(Integer)
    round = Column(Integer)
    date = Column(DateTime)
    home_team_id = Column(Integer, ForeignKey('teams.id'))
    away_team_id = Column(Integer, ForeignKey('teams.id'))
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    tournament = relationship("Tournament", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerMatchStats", back_populates="match")

class PlayerMatchStats(Base):
    __tablename__ = 'player_match_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    match_id = Column(Integer, ForeignKey('matches.id'))
    team_id = Column(Integer, ForeignKey('teams.id'))
    
    minutes = Column(Integer, default=0)
    rating = Column(Float, nullable=True)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    total_points = Column(Float, default=0.0) # Calculated Fantasy Points
    detailed_stats = Column(JSON) # Store raw stats for flexible feature eng
    
    player = relationship("Player", back_populates="match_stats")
    match = relationship("Match", back_populates="player_stats")
    team = relationship("Team")

class FantasyRules(Base):
    __tablename__ = 'fantasy_rules'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String) # e.g. "goal", "assist"
    points = Column(Float)
    position = Column(String, default="ALL") # "ALL", "G", "D", "M", "F"

# Database Initialization
DB_PATH = os.path.join(os.path.dirname(__file__), 'sofascore_fantasy.db')
engine = create_engine(f'sqlite:///{DB_PATH}')
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
    
    # Initialize default rules if empty
    session = SessionLocal()
    if session.query(FantasyRules).count() == 0:
        default_rules = [
            # General
            FantasyRules(action="rating_6.5", points=0, position="ALL"), 
            # Stats based (Approximation of SofaScore Official)
            FantasyRules(action="goal", points=7, position="D"),
            FantasyRules(action="goal", points=7, position="G"),
            FantasyRules(action="goal", points=6, position="M"),
            FantasyRules(action="goal", points=5, position="F"),
            FantasyRules(action="assist", points=3, position="ALL"),
            FantasyRules(action="clean_sheet", points=4, position="G"),
            FantasyRules(action="clean_sheet", points=4, position="D"),
            FantasyRules(action="clean_sheet", points=1, position="M"),
            FantasyRules(action="penalty_miss", points=-2, position="ALL"),
            FantasyRules(action="red_card", points=-3, position="ALL"),
            FantasyRules(action="own_goal", points=-2, position="ALL"),
        ]
        session.add_all(default_rules)
        session.commit()
    session.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
