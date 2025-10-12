"""
Fő pipeline orchestration: event archive → CSV-k frissítése.
"""

import logging
from typing import List, Set
import pandas as pd

from scrapers import EventScraper, MatchScraper, H2HScraper, TeamHistoryScraper, RankingScraper
from utils import CSVManager, URLTracker
from utils.config import *
import os

logger = logging.getLogger(__name__)


class ScrapePipeline:
    """Teljes scrape pipeline orchestration."""
    
    def __init__(self):
        self.csv_manager = CSVManager()
        self.url_tracker = URLTracker(SCRAPED_URLS_JSON)
        
        # CSV könyvtár biztosítása
        self.csv_manager.ensure_directory(EVENTS_CSV)
    
    def run_from_archive(self, archive_url: str = "https://www.hltv.org/events/archive?eventType=MAJOR"):
        """
        Fő pipeline futtatása az event archive-ból kiindulva.
        
        Args:
            archive_url: HLTV events archive URL
        """
        logger.info(f"🚀 Pipeline indítása az archive-ból")
        logger.info(f"📋 Archive URL: {archive_url}")
        
        # 0️⃣ EVENT ARCHIVE SCRAPING
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Event Archive Scraping")
        logger.info(f"{'='*60}")
        
        event_scraper = EventScraper()
        events_df = event_scraper.scrape_event_archive(archive_url)
        
        if events_df.empty:
            logger.error("❌ Nincs event az archive-ban!")
            return
        
        # Events mentése
        self.csv_manager.append_or_update(EVENTS_CSV, events_df, ['event_id'])
        logger.info(f"✅ {len(events_df)} event mentve: {EVENTS_CSV}")
        
        # Event ID-k kigyűjtése
        event_ids = events_df['event_id'].tolist()
        
        logger.info(f"🎯 {len(event_ids)} event lesz feldolgozva")
        
        # Továbblépés a feldolgozáshoz
        self._process_events(event_ids)
    
    def _process_events(self, event_ids: List[str]):
        """
        Event-ek feldolgozása (matches → H2H → team history).
        
        Args:
            event_ids: Event ID-k listája
        """
        all_match_urls = []
        
        # 1️⃣ MATCH SCRAPING minden event-hez
        for event_id in event_ids:
            try:
                # Skip ha már friss
                if self.url_tracker.is_scraped('events', event_id, URL_CACHE_DAYS):
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Event feldolgozása: {event_id}")
                logger.info(f"{'='*60}")
                
                # Match scraping
                match_scraper = MatchScraper()
                matches_df = match_scraper.scrape_event_matches(event_id)
                
                if not matches_df.empty:
                    self.csv_manager.append_or_update(RESULTS_CSV, matches_df, ['match_id'])
                    
                    # Match URL-ek gyűjtése
                    if 'link' in matches_df.columns:
                        all_match_urls.extend(matches_df['link'].tolist())
                    
                    logger.info(f"✅ {len(matches_df)} meccs mentve")
                
                # Event megjelölése scrape-eltnek
                self.url_tracker.mark_scraped('events', event_id)
                
            except Exception as e:
                logger.error(f"❌ Hiba az event feldolgozásakor ({event_id}): {e}", exc_info=True)
                continue
        
        # 2️⃣ H2H SCRAPING (minden meccshez)
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 H2H scraping: {len(all_match_urls)} meccs")
        logger.info(f"{'='*60}")
        
        for idx, match_url in enumerate(all_match_urls, 1):
            try:
                match_id = match_url.split('/')[4]
                
                if self.url_tracker.is_scraped('matches', match_id, URL_CACHE_DAYS):
                    continue
                
                logger.info(f"[{idx}/{len(all_match_urls)}] Processing match: {match_id}")
                
                h2h_scraper = H2HScraper()
                h2h_df = h2h_scraper.scrape_match_h2h(match_url)
                
                if not h2h_df.empty:
                    self.csv_manager.append_or_update(MATCHES_H2H_CSV, h2h_df, ['match_id'])
                    self.url_tracker.mark_scraped('matches', match_id)
                    logger.info(f"  ✅ H2H mentve")
                
            except Exception as e:
                logger.error(f"❌ H2H hiba ({match_url}): {e}", exc_info=True)
                continue
        
        # 3️⃣ TEAM HISTORY SCRAPING
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 Team history scraping")
        logger.info(f"{'='*60}")

        # Team ID-k beolvasása a teams.csv-ből
        import os
        if not os.path.exists(TEAMS_CSV):
            logger.warning("⚠️ Nincs teams.csv! Team history scraping kihagyva.")
        else:
            teams_df = pd.read_csv(TEAMS_CSV)
            unique_team_ids = teams_df['team_id'].dropna().unique().tolist()
            
            logger.info(f"📋 {len(unique_team_ids)} egyedi csapat található a teams.csv-ben")
            
            for idx, team_id in enumerate(sorted(unique_team_ids), 1):
                try:
                    team_id_str = str(team_id)
                    
                    if self.url_tracker.is_scraped('teams', team_id_str, URL_CACHE_DAYS):
                        continue
                    
                    # Csapatnév lekérése (ha van)
                    team_name = teams_df[teams_df['team_id'] == team_id]['team_name'].iloc[0] if len(teams_df[teams_df['team_id'] == team_id]) > 0 else "Unknown"
                    
                    logger.info(f"[{idx}/{len(unique_team_ids)}] Processing team: {team_name} ({team_id_str})")
                    
                    team_scraper = TeamHistoryScraper()
                    team_df = team_scraper.scrape_team_history(team_id_str)
                    
                    if not team_df.empty:
                        self.csv_manager.append_or_update(TEAM_HISTORY_CSV, team_df, ['team_id'])
                        self.url_tracker.mark_scraped('teams', team_id_str)
                        logger.info(f"  ✅ Team history mentve")
                    
                except Exception as e:
                    logger.error(f"❌ Team history hiba ({team_id}): {e}", exc_info=True)
                    continue
        
        # 4️⃣ RANKINGS SCRAPING (egyszer)
        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 Rankings scraping")
        logger.info(f"{'='*60}")
        
        try:
            ranking_scraper = RankingScraper()
            rankings_df = ranking_scraper.scrape_rankings()
            
            if not rankings_df.empty:
                # Rankings-nél APPEND (nem update), mert time-series
                self.csv_manager.append_or_update(RANKINGS_CSV, rankings_df, ['date', 'team_id'])
                logger.info(f"✅ {len(rankings_df)} ranking mentve")
        except Exception as e:
            logger.error(f"❌ Rankings hiba: {e}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Pipeline befejezve!")
        logger.info(f"{'='*60}")