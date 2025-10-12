"""
ML-ready scraping pipeline: target meccsek → team history → features.
"""

import logging
import pandas as pd
from typing import List
import os

from scrapers import MatchScraper, H2HScraper, TeamHistoryScraper
from utils import CSVManager, URLTracker
from .ml_config import *

logger = logging.getLogger(__name__)


class MLScrapePipeline:
    """ML-célú scraping pipeline."""
    
    def __init__(self):
        self.csv_manager = CSVManager()
        self.url_tracker = URLTracker(ML_SCRAPED_URLS_JSON)
        
        # Könyvtár biztosítása
        if not os.path.exists(ML_DATA_DIR):
            os.makedirs(ML_DATA_DIR)
            logger.info(f"📁 ML data könyvtár létrehozva: {ML_DATA_DIR}")
    
    def run(self, event_ids: List[str]):
        """
        Fő ML pipeline futtatása.
        
        Args:
            event_ids: Event ID-k listája (pl. ["7441", "7902"])
        """
        logger.info("="*80)
        logger.info("🎯 ML SCRAPING PIPELINE INDÍTÁSA")
        logger.info("="*80)
        
        # 1️⃣ TARGET MATCHES scraping
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 1. TARGET MATCHES SCRAPING")
        logger.info(f"{'='*60}")
        
        all_matches = []
        for event_id in event_ids:
            logger.info(f"\n📊 Event scraping: {event_id}")
            matches_df = self._scrape_event_matches(event_id)
            if not matches_df.empty:
                all_matches.append(matches_df)
        
        if not all_matches:
            logger.error("❌ Nincs target match!")
            return
        
        target_matches_df = pd.concat(all_matches, ignore_index=True)
        self.csv_manager.append_or_update(ML_TARGET_MATCHES_CSV, target_matches_df, ['match_id'])
        logger.info(f"✅ {len(target_matches_df)} target match mentve")
        
        # 2️⃣ MATCH DETAILS (H2H + player stats)
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 2. MATCH DETAILS SCRAPING (H2H)")
        logger.info(f"{'='*60}")
        
        match_details_list = []
        for idx, row in target_matches_df.iterrows():
            match_url = row['link']
            match_id = row['match_id']
            
            # Skip ha már scrape-elve
            if self.url_tracker.is_scraped('matches', match_id, ML_URL_CACHE_DAYS):
                logger.info(f"[{idx+1}/{len(target_matches_df)}] ⏭️  Skip match: {match_id}")
                continue
            
            logger.info(f"[{idx+1}/{len(target_matches_df)}] 🔍 Scraping match: {match_id}")
            
            h2h_df = self._scrape_match_h2h(match_url, row['event_id'])
            if not h2h_df.empty:
                match_details_list.append(h2h_df)
                self.url_tracker.mark_scraped('matches', match_id)
        
        if match_details_list:
            match_details_df = pd.concat(match_details_list, ignore_index=True)
            self.csv_manager.append_or_update(ML_MATCH_DETAILS_CSV, match_details_df, ['match_id'])
            logger.info(f"✅ {len(match_details_df)} match detail mentve")
        
        # 3️⃣ TEAM HISTORY scraping (unique team_id-k alapján)
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 3. TEAM HISTORY SCRAPING")
        logger.info(f"{'='*60}")
        
        # Unique team ID-k a match details-ből
        if os.path.exists(ML_MATCH_DETAILS_CSV):
            details_df = pd.read_csv(ML_MATCH_DETAILS_CSV)
            team_ids = set()
            
            if 'home_team_id' in details_df.columns:
                team_ids.update(details_df['home_team_id'].dropna().astype(str).tolist())
            if 'away_team_id' in details_df.columns:
                team_ids.update(details_df['away_team_id'].dropna().astype(str).tolist())
            
            logger.info(f"📋 {len(team_ids)} unique csapat található")
            
            team_history_list = []
            for idx, team_id in enumerate(sorted(team_ids), 1):
                if self.url_tracker.is_scraped('teams', team_id, ML_URL_CACHE_DAYS):
                    logger.info(f"[{idx}/{len(team_ids)}] ⏭️  Skip team: {team_id}")
                    continue
                
                logger.info(f"[{idx}/{len(team_ids)}] 📈 Scraping team: {team_id}")
                
                history_df = self._scrape_team_matches(team_id)
                if not history_df.empty:
                    team_history_list.append(history_df)
                    self.url_tracker.mark_scraped('teams', team_id)
            
            if team_history_list:
                team_history_df = pd.concat(team_history_list, ignore_index=True)
                self.csv_manager.append_or_update(ML_TEAM_HISTORY_CSV, team_history_df, ['match_id', 'team_id'])
                logger.info(f"✅ {len(team_history_df)} team history mentve")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ ML SCRAPING BEFEJEZVE!")
        logger.info(f"{'='*60}")
    
    def _scrape_event_matches(self, event_id: str) -> pd.DataFrame:
        """Event meccsek scraping."""
        match_scraper = MatchScraper(headless=False)
        return match_scraper.scrape_event_matches(event_id)
    
    def _scrape_match_h2h(self, match_url: str, event_id: str) -> pd.DataFrame:
        """Match H2H scraping."""
        h2h_scraper = H2HScraper(headless=False)
        h2h_df = h2h_scraper.scrape_match_h2h(match_url)
        
        # Event_id hozzáadása utólag
        if not h2h_df.empty and event_id:
            h2h_df['event_id'] = event_id
        
        return h2h_df
    
    def _scrape_team_matches(self, team_id: str, max_matches: int = 50) -> pd.DataFrame:
        """
        Csapat match history scraping (részletes, time-ordered).
        
        Args:
            team_id: HLTV team ID
            max_matches: Maximum hány meccset scrape-eljen
        
        Returns:
            DataFrame: team_id, match_id, match_date, opponent_id, opponent_name, 
                      result, score_for, score_against, map_type, event_id, link
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from scrapers.base_scraper import BaseScraper
        
        class TeamMatchHistoryScraper(BaseScraper):
            def scrape(self, team_id: str, max_matches: int):
                url = f"https://www.hltv.org/results?team={team_id}"
                self._init_driver()
                self.driver.get(url)
                
                wait = WebDriverWait(self.driver, 15)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-holder")))
                self._random_delay()
                
                matches = []
                results_holder = self.driver.find_element(By.CLASS_NAME, "results-holder")
                sublists = results_holder.find_elements(By.CLASS_NAME, "results-sublist")
                
                for sublist in sublists:
                    if len(matches) >= max_matches:
                        break
                    
                    # Dátum
                    headline = sublist.find_element(By.CLASS_NAME, "standard-headline").text.strip()
                    match_date = headline.replace("Results for", "").strip()
                    
                    match_blocks = sublist.find_elements(By.CLASS_NAME, "result-con")
                    for match in match_blocks:
                        if len(matches) >= max_matches:
                            break
                        
                        try:
                            a_tag = match.find_element(By.TAG_NAME, "a")
                            match_url = a_tag.get_attribute("href")
                            match_id = match_url.split('/')[4]
                            
                            table = a_tag.find_element(By.TAG_NAME, "table")
                            tds = table.find_elements(By.TAG_NAME, "td")
                            
                            team1_name = tds[0].find_element(By.CLASS_NAME, "team").text.strip()
                            team2_name = tds[2].find_element(By.CLASS_NAME, "team").text.strip()
                            
                            # Score
                            score_spans = tds[1].find_elements(By.TAG_NAME, "span")
                            score1 = int(score_spans[0].text.strip())
                            score2 = int(score_spans[1].text.strip())
                            
                            # Winner detection
                            team1_html = tds[0].get_attribute("innerHTML")
                            won = "team-won" in team1_html
                            
                            # Map type
                            try:
                                map_text = a_tag.find_element(By.CSS_SELECTOR, ".map-text").text.strip()
                            except:
                                map_text = "bo1"
                            
                            # Opponent (másik csapat)
                            opponent_name = team2_name if team1_name else team1_name
                            
                            # Event ID (megpróbáljuk kinyerni a URL-ből, de nem mindig van)
                            event_id_match = None
                            
                            matches.append({
                                "team_id": team_id,
                                "match_id": match_id,
                                "match_date": match_date,
                                "opponent_name": opponent_name,
                                "opponent_id": None,  # Ezt később ki lehet egészíteni
                                "result": "win" if won else "loss",
                                "score_for": score1,
                                "score_against": score2,
                                "map_type": map_text,
                                "event_id": event_id_match,
                                "link": match_url
                            })
                            
                        except Exception as e:
                            logger.debug(f"⚠️ Match parse hiba: {e}")
                            continue
                
                self.close()
                return pd.DataFrame(matches)
        
        scraper = TeamMatchHistoryScraper(headless=False)
        return scraper.scrape(team_id, max_matches)