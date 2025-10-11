"""
Fő pipeline orchestration: event URL-ek → CSV-k frissítése.
"""

import logging
from typing import List
import pandas as pd

from scrapers import EventScraper, MatchScraper, H2HScraper, TeamHistoryScraper, RankingScraper
from utils import CSVManager, URLTracker
from utils.config import *

logger = logging.getLogger(__name__)


class ScrapePipeline:
    """Teljes scrape pipeline orchestration."""
    
    def __init__(self):
        self.csv_manager = CSVManager()
        self.url_tracker = URLTracker(SCRAPED_URLS_JSON)
        
        # CSV könyvtár biztosítása
        self.csv_manager.ensure_directory(EVENTS_CSV)
    
    def run(self, event_urls: List[str]):
        """
        Fő pipeline futtatása.
        
        Args:
            event_urls: HLTV event URL-ek listája
        """
        logger.info(f"🚀 Pipeline indítása {len(event_urls)} event-tel")
        
        all_match_urls = []
        all_team_ids = set()
        
        # 1️⃣ EVENT SCRAPING
        for event_url in event_urls:
            try:
                event_id = event_url.split('/')[4]
                
                # Skip ha már friss
                if self.url_tracker.is_scraped('events', event_id, URL_CACHE_DAYS):
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Event feldolgozása: {event_id}")
                logger.info(f"{'='*60}")
                
                # Event metadata scrape
                event_scraper = EventScraper()
                event_df = event_scraper.scrape_event(event_url)
                
                if not event_df.empty:
                    self.csv_manager.append_or_update(EVENTS_CSV, event_df, ['event_id'])
                    self.url_tracker.mark_scraped('events', event_id)
                
                # 2️⃣ MATCH SCRAPING (az event-hez)
                match_scraper = MatchScraper()
                matches_df = match_scraper.scrape_event_matches(event_id)
                
                if not matches_df.empty:
                    self.csv_manager.append_or_update(RESULTS_CSV, matches_df, ['link'])
                    
                    # Match URL-ek gyűjtése
                    if 'link' in matches_df.columns:
                        all_match_urls.extend(matches_df['link'].tolist())
                    
                    # Team ID-k gyűjtése
                    if 'team_home' in matches_df.columns and 'team_away' in matches_df.columns:
                        # FONTOS: Itt team_id kell, nem team_name!
                        # A scraperben add hozzá a team_id-kat is!
                        pass
                
            except Exception as e:
                logger.error(f"❌ Hiba az event feldolgozásakor ({event_url}): {e}", exc_info=True)
                continue
        
        # 3️⃣ H2H SCRAPING (minden meccshez)
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 H2H scraping: {len(all_match_urls)} meccs")
        logger.info(f"{'='*60}")
        
        for match_url in all_match_urls:
            try:
                match_id = match_url.split('/')[4]
                
                if self.url_tracker.is_scraped('matches', match_id, URL_CACHE_DAYS):
                    continue
                
                h2h_scraper = H2HScraper()
                h2h_df = h2h_scraper.scrape_match_h2h(match_url)
                
                if not h2h_df.empty:
                    self.csv_manager.append_or_update(MATCHES_H2H_CSV, h2h_df, ['match_id'])
                    self.url_tracker.mark_scraped('matches', match_id)
                
            except Exception as e:
                logger.error(f"❌ H2H hiba ({match_url}): {e}", exc_info=True)
                continue
        
        # 4️⃣ TEAM HISTORY SCRAPING
        # FONTOS: Gyűjtsd ki a team_id-kat a results.csv-ből!
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 Team history scraping: {len(all_team_ids)} csapat")
        logger.info(f"{'='*60}")
        
        for team_id in all_team_ids:
            try:
                if self.url_tracker.is_scraped('teams', team_id, URL_CACHE_DAYS):
                    continue
                
                team_scraper = TeamHistoryScraper()
                team_df = team_scraper.scrape_team_history(team_id)
                
                if not team_df.empty:
                    self.csv_manager.append_or_update(TEAM_HISTORY_CSV, team_df, ['team_id'])
                    self.url_tracker.mark_scraped('teams', team_id)
                
            except Exception as e:
                logger.error(f"❌ Team history hiba ({team_id}): {e}", exc_info=True)
                continue
        
        # 5️⃣ RANKINGS SCRAPING (egyszer)
        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 Rankings scraping")
        logger.info(f"{'='*60}")
        
        try:
            ranking_scraper = RankingScraper()
            rankings_df = ranking_scraper.scrape_rankings()
            
            if not rankings_df.empty:
                # Rankings-nél APPEND (nem update), mert time-series
                self.csv_manager.append_or_update(RANKINGS_CSV, rankings_df, ['date', 'team_id'])
        except Exception as e:
            logger.error(f"❌ Rankings hiba: {e}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Pipeline befejezve!")
        logger.info(f"{'='*60}")