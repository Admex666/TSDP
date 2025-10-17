"""
Main orchestrator for CS:GO scraper pipeline
"""
import sys
from typing import List
import pandas as pd
import re
from datetime import timedelta

from config.settings import N_HISTORICAL_MATCHES
from database.db_manager import DatabaseManager
from logger.scrape_logger import ScrapeLogger
from scrapers.match_scraper import MatchScraper
from scrapers.h2h_scraper import H2HScraper
from scrapers.team_history_scraper import TeamHistoryScraper
from processors.feature_builder import FeatureBuilder


class ScraperOrchestrator:
    def __init__(self, headless: bool = True):
        self.db = DatabaseManager()
        self.db.create_tables()
        self.logger = ScrapeLogger("Orchestrator")
        self.headless = headless
        self.feature_builder = FeatureBuilder(self.db)
    
    def scrape_event(self, event_id: str, event_name: str = None, 
                    build_features: bool = True):
        """
        Complete event scraping pipeline
        
        Args:
            event_id: HLTV event ID
            event_name: Event name (optional)
            build_features: Whether to build ML features after scraping
        """
        self.logger.info(f"{'='*60}")
        self.logger.info(f"🎯 STARTING EVENT SCRAPE: {event_id}")
        self.logger.info(f"{'='*60}")
        
        # Save event
        if event_name:
            self.db.save_event(event_id, event_name)
        
        # Step 1: Scrape target matches
        matches_df = self._scrape_target_matches(event_id)
        if matches_df.empty:
            self.logger.error(f"No matches found for event {event_id}")
            return
        
        self.logger.info(f"✅ Found {len(matches_df)} matches to process")
        
        # Step 2: Process each match
        for idx, match_row in matches_df.iterrows():
            self.logger.progress(idx + 1, len(matches_df), "matches")
            
            try:
                self._process_match(match_row, event_id)
            except Exception as e:
                self.logger.error(f"Failed to process match {match_row['match_id']}: {e}")
                continue
        
        # Step 3: Build ML features if requested
        if build_features:
            self.logger.info(f"\n🔨 Building ML features...")
            self._build_features_for_event(event_id)
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"✅ EVENT SCRAPE COMPLETE: {event_id}")
        self.logger.info(f"{'='*60}")
    
    def _scrape_target_matches(self, event_id: str) -> pd.DataFrame:
        """Scrape all matches for an event"""
        self.logger.scrape_start("EVENT_MATCHES", event_id)
        
        url = f"https://www.hltv.org/results?event={event_id}"
        
        # Check if already scraped
        if self.db.is_scraped(url, hours=24):
            self.logger.scrape_skip("EVENT_MATCHES", event_id, "Already scraped in last 24h")
            return self.db.get_event_matches(event_id)
        
        try:
            scraper = MatchScraper(headless=self.headless)
            matches_df = scraper.scrape_event_matches(event_id)
            
            if not matches_df.empty:
                # Clean dates
                matches_df["date_clean"] = matches_df["date"].apply(
                    lambda x: re.sub(r'(\d+)(st|nd|rd|th)', r'\1', x)
                )
                matches_df["date_parsed"] = pd.to_datetime(
                    matches_df["date_clean"], format="%B %d %Y"
                )
                
                # Save to database
                self.db.save_matches(matches_df, event_id)
                self.db.log_scrape(url, "event_matches", success=True, event_id=event_id)
                self.logger.scrape_success("EVENT_MATCHES", event_id, len(matches_df))
            
            return matches_df
            
        except Exception as e:
            self.db.log_scrape(url, "event_matches", success=False, 
                             error_msg=str(e), event_id=event_id)
            self.logger.scrape_error("EVENT_MATCHES", event_id, str(e))
            return pd.DataFrame()
    
    def _process_match(self, match_row: pd.Series, event_id: str):
        """Process a single match: scrape H2H and team histories"""
        match_id = match_row['match_id']
        match_url = match_row['link']
        match_date = match_row['date_parsed']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎮 Processing: {match_row['team_home']} vs {match_row['team_away']}")
        self.logger.info(f"   Match ID: {match_id}")
        self.logger.info(f"   Date: {match_date.date()}")
        
        # 1. Scrape match H2H
        h2h_data = self._scrape_match_h2h(match_url, match_id)
        if not h2h_data:
            self.logger.warning(f"No H2H data for match {match_id}, skipping")
            return
        
        # 2. Extract team info
        teams = {
            'home': {
                'team_id': str(h2h_data.get('home_team_id', 'unknown')),
                'team_name': h2h_data['home_team']
            },
            'away': {
                'team_id': str(h2h_data.get('away_team_id', 'unknown')),
                'team_name': h2h_data['away_team']
            }
        }
        
        # 3. Scrape team histories and their H2H data
        for side in ['home', 'away']:
            team_id = teams[side]['team_id']
            team_name = teams[side]['team_name']
            
            self.logger.info(f"\n📊 Processing {side.upper()}: {team_name}")
            
            # Scrape team history
            history_df = self._scrape_team_history(team_id, team_name, match_date)
            
            if history_df.empty or len(history_df) < 3:
                self.logger.warning(f"Insufficient history for {team_name}")
                continue
            
            # Scrape H2H for last N matches
            self._scrape_historical_h2h(team_id, team_name, history_df, 
                                       match_date, N_HISTORICAL_MATCHES)
        
        self.logger.info(f"✅ Match {match_id} processing complete")
    
    def _scrape_match_h2h(self, match_url: str, match_id: str) -> dict:
        """Scrape H2H data for a match"""
        self.logger.scrape_start("MATCH_H2H", match_id)
        
        # Check if already scraped
        #if self.db.is_scraped(match_url, hours=24):
        #    existing_h2h = self.db.get_h2h(match_id)
        #    if existing_h2h:
        #        self.logger.scrape_skip("MATCH_H2H", match_id, "Already in database")
        #        return existing_h2h
        
        # DEBUG: Mindig scrapeljünk
        self.logger.info(f"🔍 DEBUG: Mindig scrapeljük a H2H-t, cache kikapcsolva")
        
        try:
            scraper = H2HScraper(headless=self.headless)
            h2h_df = scraper.scrape_match_h2h(match_url)
            
            self.logger.info(f"🔍 DEBUG: H2H scraping result - Üres: {h2h_df.empty}, Sorok: {len(h2h_df)}")
            
            if not h2h_df.empty:
                h2h_data = h2h_df.iloc[0].to_dict()
                self.db.save_h2h(match_id, h2h_data)
                self.db.log_scrape(match_url, "match_h2h", success=True, match_id=match_id)
                self.logger.scrape_success("MATCH_H2H", match_id)
                return h2h_data
            else:
                self.logger.warning(f"🔍 DEBUG: H2H DataFrame üres")
            
        except Exception as e:
            self.db.log_scrape(match_url, "match_h2h", success=False, 
                            error_msg=str(e), match_id=match_id)
            self.logger.scrape_error("MATCT_H2H", match_id, str(e))
        
        return None
    
    def _scrape_team_history(self, team_id: str, team_name: str, 
                            before_date: pd.Timestamp) -> pd.DataFrame:
        """Scrape team match history"""
        self.logger.scrape_start("TEAM_HISTORY", team_name)
        
        # Check database first
        history_df = self.db.get_team_history(team_id, before_date=before_date, limit=100)
        
        if not history_df.empty and len(history_df) >= 20:
            self.logger.scrape_skip("TEAM_HISTORY", team_name, "Sufficient data in database")
            return history_df
        
        url = f"https://www.hltv.org/results?team={team_id}"
        
        try:
            scraper = TeamHistoryScraper(headless=self.headless)
            history_df = scraper.scrape_team_matches(team_id, max_matches=100)
            
            if not history_df.empty:
                # Filter before match date
                history_df = history_df[history_df['date_parsed'] < before_date]
                
                # Save to database
                self.db.save_team_history(team_id, history_df)
                self.db.log_scrape(url, "team_history", success=True, team_id=team_id)
                self.logger.scrape_success("TEAM_HISTORY", team_name, len(history_df))
            
            return history_df
            
        except Exception as e:
            self.db.log_scrape(url, "team_history", success=False, 
                             error_msg=str(e), team_id=team_id)
            self.logger.scrape_error("TEAM_HISTORY", team_name, str(e))
            return pd.DataFrame()
    
    def _scrape_historical_h2h(self, team_id: str, team_name: str, 
                              history_df: pd.DataFrame, match_date: pd.Timestamp,
                              n_matches: int):
        """Scrape H2H data for team's last N matches"""
        self.logger.info(f"🔍 Scraping H2H for last {n_matches} matches of {team_name}")
        
        last_n = history_df.head(n_matches)
        
        for idx, hist_match in last_n.iterrows():
            hist_match_id = hist_match['match_id']
            hist_match_url = hist_match['link']
            
            # Check if already scraped
            existing = self.db.get_h2h(hist_match_id)
            if existing:
                self.logger.scrape_skip("HIST_H2H", hist_match_id, "Already in database")
                continue
            
            self.logger.info(f"  [{idx+1}/{n_matches}] {hist_match['date_parsed'].date()} vs {hist_match['opponent_name']}")
            
            try:
                scraper = H2HScraper(headless=self.headless)
                h2h_df = scraper.scrape_match_h2h(hist_match_url)
                
                if not h2h_df.empty:
                    h2h_data = h2h_df.iloc[0].to_dict()
                    self.db.save_h2h(hist_match_id, h2h_data)
                    self.db.log_scrape(hist_match_url, "match_h2h", 
                                     success=True, match_id=hist_match_id)
                    self.logger.scrape_success("HIST_H2H", hist_match_id)
                
            except Exception as e:
                self.db.log_scrape(hist_match_url, "match_h2h", 
                                 success=False, error_msg=str(e), match_id=hist_match_id)
                self.logger.scrape_error("HIST_H2H", hist_match_id, str(e))
                continue
    
    def _build_features_for_event(self, event_id: str):
        """Build ML features for all matches in an event"""
        matches_df = self.db.get_event_matches(event_id)
        
        for idx, match in matches_df.iterrows():
            match_id = match['match_id']
            
            try:
                features = self.feature_builder.build_ml_input(match_id)
                if features:
                    self.db.save_ml_features(match_id, features)
                    self.logger.info(f"✅ Features saved for match {match_id}")
                else:
                    self.logger.warning(f"⚠️ Could not build features for match {match_id}")
            
            except Exception as e:
                self.logger.error(f"Error building features for {match_id}: {e}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <event_id> [event_name]")
        print("Example: python main.py 8292 'ESL Pro League Season 21'")
        sys.exit(1)
    
    event_id = sys.argv[1]
    event_name = sys.argv[2] if len(sys.argv) > 2 else None

    orchestrator = ScraperOrchestrator(headless=False)
    orchestrator.scrape_event(event_id, event_name, build_features=True)
    
if __name__ == "__main__":
    main()