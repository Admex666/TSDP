"""
Database manager for CS:GO scraper
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from config.settings import DB_URL
from database.models import Base, ScrapeLog, Event, Match, MatchH2H, TeamHistory, Ranking, Odds, MLFeature


class DatabaseManager:
    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url, echo=False)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # =====================
    # SCRAPE LOG
    # =====================
    
    def log_scrape(self, url: str, scrape_type: str, success: bool = True, 
                   error_msg: str = None, event_id: str = None, 
                   match_id: str = None, team_id: str = None):
        """Log a scrape attempt"""
        with self.session_scope() as session:
            log = ScrapeLog(
                url=url,
                scrape_type=scrape_type,
                success=success,
                error_msg=error_msg,
                event_id=event_id,
                match_id=match_id,
                team_id=team_id
            )
            session.add(log)
    
    def is_scraped(self, url: str, hours: int = 24) -> bool:
        """Check if URL was successfully scraped in the last N hours"""
        with self.session_scope() as session:
            cutoff = datetime.utcnow() - pd.Timedelta(hours=hours)
            log = session.query(ScrapeLog).filter(
                ScrapeLog.url == url,
                ScrapeLog.success == True,
                ScrapeLog.timestamp > cutoff
            ).first()
            return log is not None
    
    # =====================
    # EVENTS
    # =====================
    
    def save_event(self, event_id: str, event_name: str, url: str = None):
        """Save or update event"""
        with self.session_scope() as session:
            event = session.query(Event).filter_by(event_id=event_id).first()
            if not event:
                event = Event(event_id=event_id, event_name=event_name, url=url)
                session.add(event)
            else:
                event.event_name = event_name
                if url:
                    event.url = url
    
    # =====================
    # MATCHES
    # =====================
    
    def save_matches(self, matches_df: pd.DataFrame, event_id: str):
        """Save matches from DataFrame"""
        with self.session_scope() as session:
            for _, row in matches_df.iterrows():
                match = session.query(Match).filter_by(match_id=row['match_id']).first()
                if not match:
                    match = Match(
                        match_id=row['match_id'],
                        event_id=event_id,
                        date=row['date_parsed'],
                        date_month=row.get('date_month'),
                        date_day=row.get('date_day'),
                        team_home=row['team_home'],
                        team_home_id=row.get('team_home_id'),
                        team_away=row['team_away'],
                        team_away_id=row.get('team_away_id'),
                        score_home=row.get('score_home'),
                        score_away=row.get('score_away'),
                        rounds=row.get('rounds'),
                        url=row['link']
                    )
                    session.add(match)
    
    def get_match(self, match_id: str) -> Optional[Dict]:
        """Get match by ID"""
        with self.session_scope() as session:
            match = session.query(Match).filter_by(match_id=match_id).first()
            if match:
                return {
                    'match_id': match.match_id,
                    'event_id': match.event_id,
                    'date': match.date,
                    'team_home': match.team_home,
                    'team_away': match.team_away,
                    'score_home': match.score_home,
                    'score_away': match.score_away,
                    'url': match.url
                }
            return None
    
    def get_event_matches(self, event_id: str) -> pd.DataFrame:
        """Get all matches for an event"""
        with self.session_scope() as session:
            matches = session.query(Match).filter_by(event_id=event_id).all()
            return pd.DataFrame([{
                'match_id': m.match_id,
                'date': m.date,
                'team_home': m.team_home,
                'team_away': m.team_away,
                'score_home': m.score_home,
                'score_away': m.score_away,
                'url': m.url
            } for m in matches])
    
    # =====================
    # MATCH H2H
    # =====================
    
    def save_h2h(self, match_id: str, h2h_data: Dict):
        """Save match H2H data"""
        with self.session_scope() as session:
            h2h = session.query(MatchH2H).filter_by(match_id=match_id).first()
            if not h2h:
                h2h = MatchH2H(match_id=match_id, **h2h_data)
                session.add(h2h)
            else:
                for key, value in h2h_data.items():
                    setattr(h2h, key, value)
    
    def get_h2h(self, match_id: str) -> Optional[Dict]:
        """Get H2H data for a match"""
        with self.session_scope() as session:
            h2h = session.query(MatchH2H).filter_by(match_id=match_id).first()
            if h2h:
                return {c.name: getattr(h2h, c.name) for c in h2h.__table__.columns}
            return None
    
    # =====================
    # TEAM HISTORY
    # =====================
    
    def save_team_history(self, team_id: str, history_df: pd.DataFrame):
        """Save team match history"""
        with self.session_scope() as session:
            for _, row in history_df.iterrows():
                # Check if already exists
                exists = session.query(TeamHistory).filter_by(
                    team_id=team_id,
                    match_id=row['match_id']
                ).first()
                
                if not exists:
                    history = TeamHistory(
                        team_id=team_id,
                        match_id=row['match_id'],
                        match_date=row['date_parsed'],
                        opponent_name=row['opponent_name'],
                        result=row['result'],
                        score_for=row['score_for'],
                        score_against=row['score_against'],
                        map_type=row.get('map_type'),
                        url=row['link']
                    )
                    session.add(history)
    
    def get_team_history(self, team_id: str, before_date: datetime = None, 
                        limit: int = 100) -> pd.DataFrame:
        """Get team match history"""
        with self.session_scope() as session:
            query = session.query(TeamHistory).filter_by(team_id=team_id)
            
            if before_date:
                query = query.filter(TeamHistory.match_date < before_date)
            
            query = query.order_by(TeamHistory.match_date.desc()).limit(limit)
            
            matches = query.all()
            return pd.DataFrame([{
                'team_id': m.team_id,
                'match_id': m.match_id,
                'date_parsed': m.match_date,
                'opponent_name': m.opponent_name,
                'result': m.result,
                'score_for': m.score_for,
                'score_against': m.score_against,
                'map_type': m.map_type,
                'link': m.url
            } for m in matches])
    
    # =====================
    # RANKINGS
    # =====================
    
    def save_rankings(self, rankings_df: pd.DataFrame):
        """Save rankings data"""
        with self.session_scope() as session:
            for _, row in rankings_df.iterrows():
                # Check if already exists
                exists = session.query(Ranking).filter_by(
                    date=row['date'],
                    team_id=row['team_id']
                ).first()
                
                if not exists:
                    ranking = Ranking(
                        date=row['date'],
                        rank=row['rank'],
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        points=row['points'],
                        profile_link=row.get('profile_link')
                    )
                    session.add(ranking)
    
    def get_rankings(self, before_date: datetime = None) -> pd.DataFrame:
        """Get rankings"""
        with self.session_scope() as session:
            query = session.query(Ranking)
            
            if before_date:
                query = query.filter(Ranking.date < before_date)
            
            rankings = query.all()
            return pd.DataFrame([{
                'date': r.date,
                'rank': r.rank,
                'team_id': r.team_id,
                'team_name': r.team_name,
                'points': r.points
            } for r in rankings])
    
    # =====================
    # ML FEATURES
    # =====================
    
    def save_ml_features(self, match_id: str, features: Dict):
        """Save ML features for a match"""
        with self.session_scope() as session:
            ml_feature = session.query(MLFeature).filter_by(match_id=match_id).first()
            if not ml_feature:
                ml_feature = MLFeature(match_id=match_id, **features)
                session.add(ml_feature)
            else:
                for key, value in features.items():
                    setattr(ml_feature, key, value)
    
    def get_ml_features(self, match_id: str) -> Optional[Dict]:
        """Get ML features for a match"""
        with self.session_scope() as session:
            features = session.query(MLFeature).filter_by(match_id=match_id).first()
            if features:
                return {c.name: getattr(features, c.name) for c in features.__table__.columns}
            return None