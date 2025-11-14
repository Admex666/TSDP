"""
Teljes ML input flow teszt: Event ID → ML input sor.
Végigprinteli az egész folyamatot.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from scrapers import MatchScraper, H2HScraper
from scrapers.base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MLInputGenerator:
    """Generál egy ML input sort teljes flow-val."""
    
    def __init__(self):
        self.event_id = None
        self.target_matches = None
        self.selected_match = None
        self.match_h2h = None
        self.teams = None
        self.team_histories = {}
        self.historical_h2h = {}
        self.rankings = None
        self.ml_input_row = {}
    
    def run(self, event_id: str, match_index: int = 0):
        """
        Teljes flow futtatása.
        
        Args:
            event_id: Event ID (pl. "7441")
            match_index: Hanyadik meccset használjuk (0 = első)
        """
        self.event_id = event_id
        
        logger.info("="*80)
        logger.info("🚀 ML INPUT GENERATION - TELJES FOLYAMAT")
        logger.info("="*80)
        
        # 1️⃣ Target matches scraping
        self._step1_scrape_target_matches()
        
        # 2️⃣ Select one match
        self._step2_select_match(match_index)
        
        # 3️⃣ Match H2H scraping
        self._step3_scrape_match_h2h()
        
        # 4️⃣ Extract teams
        self._step4_extract_teams()
        
        # 5️⃣ Team match history scraping
        self._step5_scrape_team_histories()
        
        # 6️⃣ Historical matches H2H scraping (last N)
        self._step6_scrape_historical_h2h()
        
        # 7️⃣ Feature engineering (rolling features)
        self._step7_compute_rolling_features()
        
        # 8️⃣ Rankings (mock data for now)
        self._step8_add_rankings()
        
        # 9️⃣ Build final ML input row
        self._step9_build_ml_input()
        
        logger.info("\n" + "="*80)
        logger.info("✅ ML INPUT ROW KÉSZ!")
        logger.info("="*80)
        
        return self.ml_input_row
    
    def _step1_scrape_target_matches(self):
        """1️⃣ Target matches scraping."""
        logger.info("\n" + "="*60)
        logger.info("1️⃣ TARGET MATCHES SCRAPING")
        logger.info("="*60)
        logger.info(f"Event ID: {self.event_id}")
        
        scraper = MatchScraper(headless=False)
        self.target_matches = scraper.scrape_event_matches(self.event_id)
        
        if self.target_matches.empty:
            raise ValueError("❌ Nincs target match!")
        
        logger.info(f"✅ {len(self.target_matches)} meccs találva")
        logger.info(f"\n📋 Első 3 meccs:")
        print(self.target_matches[['match_id', 'team_home', 'team_away', 'score_home', 'score_away']].head(3))
    
    def _step2_select_match(self, match_index: int):
        """2️⃣ Select one match."""
        logger.info("\n" + "="*60)
        logger.info("2️⃣ MATCH KIVÁLASZTÁSA")
        logger.info("="*60)
        
        if match_index >= len(self.target_matches):
            match_index = 0
            logger.warning(f"⚠️ Index túl nagy, első meccset használom")
        
        self.selected_match = self.target_matches.iloc[match_index]
        
        logger.info(f"🎯 Kiválasztott meccs (index={match_index}):")
        logger.info(f"  Match ID:   {self.selected_match['match_id']}")
        logger.info(f"  Date:       {self.selected_match['date']}")
        logger.info(f"  Teams:      {self.selected_match['team_home']} vs {self.selected_match['team_away']}")
        logger.info(f"  Score:      {self.selected_match['score_home']} - {self.selected_match['score_away']}")
        logger.info(f"  URL:        {self.selected_match['link']}")
    
    def _step3_scrape_match_h2h(self):
        """3️⃣ Match H2H scraping."""
        logger.info("\n" + "="*60)
        logger.info("3️⃣ MATCH H2H SCRAPING")
        logger.info("="*60)
        
        match_url = self.selected_match['link']
        logger.info(f"🔍 Scraping: {match_url}")
        
        scraper = H2HScraper(headless=False)
        h2h_df = scraper.scrape_match_h2h(match_url)
        
        if h2h_df.empty:
            raise ValueError("❌ H2H scraping sikertelen!")
        
        self.match_h2h = h2h_df.iloc[0]
        
        logger.info(f"✅ H2H data:")
        logger.info(f"  Home team:        {self.match_h2h['home_team']} (ID: {self.match_h2h.get('home_team_id', 'N/A')})")
        logger.info(f"  Away team:        {self.match_h2h['away_team']} (ID: {self.match_h2h.get('away_team_id', 'N/A')})")
        logger.info(f"  H2H wins home:    {self.match_h2h['wins_home']}")
        logger.info(f"  H2H wins away:    {self.match_h2h['wins_away']}")
        logger.info(f"  Home win rate:    {self.match_h2h['home_win_rate']}")
        logger.info(f"  Home avg rating:  {self.match_h2h['home_team_avg_rating']}")
        logger.info(f"  Away avg rating:  {self.match_h2h['away_team_avg_rating']}")
    
    def _step4_extract_teams(self):
        """4️⃣ Extract teams."""
        logger.info("\n" + "="*60)
        logger.info("4️⃣ TEAMS EXTRACTION")
        logger.info("="*60)
        
        self.teams = {
            'home': {
                'team_id': str(self.match_h2h.get('home_team_id', 'unknown')),
                'team_name': self.match_h2h['home_team']
            },
            'away': {
                'team_id': str(self.match_h2h.get('away_team_id', 'unknown')),
                'team_name': self.match_h2h['away_team']
            }
        }
        
        logger.info(f"✅ Teams extracted:")
        logger.info(f"  Home: {self.teams['home']['team_name']} (ID: {self.teams['home']['team_id']})")
        logger.info(f"  Away: {self.teams['away']['team_name']} (ID: {self.teams['away']['team_id']})")
    
    def _step5_scrape_team_histories(self):
        """5️⃣ Team match history scraping."""
        logger.info("\n" + "="*60)
        logger.info("5️⃣ TEAM MATCH HISTORIES SCRAPING")
        logger.info("="*60)
        
        for side in ['home', 'away']:
            team_id = self.teams[side]['team_id']
            team_name = self.teams[side]['team_name']
            
            logger.info(f"\n📈 Scraping history: {team_name} (ID: {team_id})")
            
            history_df = self._scrape_team_matches(team_id, max_matches=100)
            
            if history_df.empty:
                logger.warning(f"⚠️ Nincs history adat: {team_name}")
                self.team_histories[side] = pd.DataFrame()
                continue
            
            self.team_histories[side] = history_df
            
            logger.info(f"✅ {len(history_df)} meccs találva")
            logger.info(f"\n📋 Utolsó 3 meccs:")
            print(history_df[['match_date', 'opponent_name', 'result', 'score_for', 'score_against']].head(3))
    
    def _scrape_team_matches(self, team_id: str, max_matches: int = 100) -> pd.DataFrame:
        """Team match history scraping (helper)."""

        class TeamHistoryScraper(BaseScraper):
            def scrape(self, team_id: str, max_matches: int):
                url = f"https://www.hltv.org/results?team={team_id}"
                self._init_driver()
                self.driver.get(url)

                wait = WebDriverWait(self.driver, 20)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-holder")))

                self._random_delay()

                matches = []
                all_sublists = self.driver.find_elements(By.CLASS_NAME, "results-sublist")
                print(f"🔍 Összesen {len(all_sublists)} results-sublist betöltve a DOM-ban.")

                for sublist in all_sublists:
                    if len(matches) >= max_matches:
                        break

                    # 🧠 VÁRJUK MEG, hogy a headline is betöltődjön
                    try:
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "standard-headline")))
                        headline = sublist.find_element(By.CLASS_NAME, "standard-headline").text.strip()
                        match_date = headline.replace("Results for", "").strip()
                    except:
                        match_date = None

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

                            score_spans = tds[1].find_elements(By.TAG_NAME, "span")
                            score1 = int(score_spans[0].text.strip())
                            score2 = int(score_spans[1].text.strip())

                            team1_html = tds[0].get_attribute("innerHTML")
                            won = "team-won" in team1_html

                            try:
                                map_text = a_tag.find_element(By.CSS_SELECTOR, ".map-text").text.strip()
                            except:
                                map_text = "bo1"

                            opponent_name = team2_name if team1_name else team1_name

                            matches.append({
                                "team_id": team_id,
                                "match_id": match_id,
                                "match_date": match_date,
                                "opponent_name": opponent_name,
                                "result": "win" if won else "loss",
                                "score_for": score1,
                                "score_against": score2,
                                "map_type": map_text,
                                "link": match_url
                            })

                        except Exception as e:
                            print(f"Hiba egy meccs feldolgozásánál: {e}")
                            continue

                self.close()
                return pd.DataFrame(matches)

        scraper = TeamHistoryScraper(headless=False)
        return scraper.scrape(team_id, max_matches)

    
    def _step6_scrape_historical_h2h(self):
        """6️⃣ Historical matches H2H scraping (last N)."""
        logger.info("\n" + "="*60)
        logger.info("6️⃣ HISTORICAL H2H SCRAPING")
        logger.info("="*60)
        
        N_MATCHES = 3  # Last N meccs H2H-ját scrape-eljük
        
        for side in ['home', 'away']:
            team_name = self.teams[side]['team_name']
            logger.info(f"\n🔍 {team_name} last {N_MATCHES} matches H2H scraping:")
            
            history = self.team_histories.get(side, pd.DataFrame())
            
            if history.empty:
                logger.warning(f"  ⚠️ Nincs history, skip")
                self.historical_h2h[side] = {
                    'avg_rating': None,
                    'avg_adr': None,
                    'avg_swing': None
                }
                continue
            
            # Last N match URL-ek
            last_n_matches = history.head(N_MATCHES)
            
            ratings = []
            adrs = []
            swings = []
            
            for idx, match in last_n_matches.iterrows():
                match_url = match['link']
                logger.info(f"  [{idx+1}/{N_MATCHES}] Scraping: {match['match_date']} vs {match['opponent_name']}")
                
                try:
                    # H2H scraping
                    h2h_scraper = H2HScraper(headless=False)
                    h2h_df = h2h_scraper.scrape_match_h2h(match_url)
                    
                    if h2h_df.empty:
                        logger.warning(f"    ⚠️ Empty H2H")
                        continue
                    
                    h2h_data = h2h_df.iloc[0]
                    
                    # Melyik csapat a mi csapatunk?
                    # A history-ban team1 az érintett csapat (mivel team_id alapján szűrtük)
                    # Ezért home_team a mi csapatunk
                    team_rating = h2h_data.get('home_team_avg_rating')
                    team_adr = h2h_data.get('home_team_avg_ADR')
                    team_swing = h2h_data.get('home_team_avg_Swing')
                    
                    if pd.notna(team_rating):
                        ratings.append(team_rating)
                    if pd.notna(team_adr):
                        adrs.append(team_adr)
                    if pd.notna(team_swing):
                        swings.append(team_swing)
                    
                    logger.info(f"    ✅ Rating: {team_rating}, ADR: {team_adr}, Swing: {team_swing}")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ H2H scraping hiba: {e}")
                    continue
            
            # Átlagok számítása
            avg_rating = np.mean(ratings) if ratings else None
            avg_adr = np.mean(adrs) if adrs else None
            avg_swing = np.mean(swings) if swings else None
            
            self.historical_h2h[side] = {
                'avg_rating': avg_rating,
                'avg_adr': avg_adr,
                'avg_swing': avg_swing,
                'n_matches_scraped': len(ratings)
            }
        
        logger.info(f"\n  📊 {team_name} historical H2H stats (last {len(ratings)}/{N_MATCHES} matches):")
        logger.info(f"    Avg rating:  {avg_rating:.4f}" if avg_rating else "    Avg rating:  N/A")
        logger.info(f"    Avg ADR:     {avg_adr:.2f}" if avg_adr else "    Avg ADR:     N/A")
        logger.info(f"    Avg Swing:   {avg_swing:.2f}" if avg_swing else "    Avg Swing:   N/A")
    
    def _step7_compute_rolling_features(self):
        """7️⃣ Feature engineering: rolling features."""
        logger.info("\n" + "="*60)
        logger.info("7️⃣ ROLLING FEATURES SZÁMÍTÁSA")
        logger.info("="*60)
        
        match_date_str = self.selected_match['date']
        # Convert to datetime (simplified)
        match_date = pd.Timestamp.now()  # Mock - production-ban parse-old a dátumot
        
        for side in ['home', 'away']:
            team_name = self.teams[side]['team_name']
            logger.info(f"\n📊 {team_name} rolling features:")
            
            history = self.team_histories.get(side, pd.DataFrame())
            
            if history.empty:
                logger.warning(f"  ⚠️ Nincs history adat")
                continue
            
            # Last 3 winrate
            last_3 = history.head(3)
            last_3_wr = (last_3['result'] == 'win').mean() if len(last_3) > 0 else None
            
            # Last 5 winrate
            last_5 = history.head(5)
            last_5_wr = (last_5['result'] == 'win').mean() if len(last_5) > 0 else None
            
            # Avg scores
            last_3_avg_for = last_3['score_for'].mean() if len(last_3) > 0 else None
            last_3_avg_against = last_3['score_against'].mean() if len(last_3) > 0 else None
            
            # Current streak
            streak = 0
            if len(history) > 0:
                last_result = history.iloc[0]['result']
                for _, match in history.iterrows():
                    if match['result'] == last_result:
                        streak += 1
                    else:
                        break
                if last_result == 'loss':
                    streak *= -1
            
            logger.info(f"  Last 3 winrate:    {last_3_wr:.3f}" if last_3_wr else "  Last 3 winrate:    N/A")
            logger.info(f"  Last 5 winrate:    {last_5_wr:.3f}" if last_5_wr else "  Last 5 winrate:    N/A")
            logger.info(f"  Last 3 avg score:  {last_3_avg_for:.1f} - {last_3_avg_against:.1f}" if last_3_avg_for else "  Last 3 avg score:  N/A")
            logger.info(f"  Current streak:    {streak:+d}")
            
            # Store
            self.ml_input_row[f'{side}_last_3_winrate'] = last_3_wr
            self.ml_input_row[f'{side}_last_5_winrate'] = last_5_wr
            self.ml_input_row[f'{side}_last_3_avg_score_for'] = last_3_avg_for
            self.ml_input_row[f'{side}_last_3_avg_score_against'] = last_3_avg_against
            self.ml_input_row[f'{side}_current_streak'] = streak
        
        # Difference features
        if 'home_last_3_winrate' in self.ml_input_row and 'away_last_3_winrate' in self.ml_input_row:
            diff_last3_wr = self.ml_input_row['home_last_3_winrate'] - self.ml_input_row['away_last_3_winrate']
            self.ml_input_row['diff_last_3_winrate'] = diff_last3_wr
            logger.info(f"\n📊 Difference features:")
            logger.info(f"  Diff last 3 WR:    {diff_last3_wr:+.3f}")
    
    def _step8_add_rankings(self):
        """8️⃣ Rankings (mock data)."""
        logger.info("\n" + "="*60)
        logger.info("8️⃣ RANKINGS")
        logger.info("="*60)
        logger.info("⚠️ MOCK DATA (production-ban rankings scraper)")
        
        # Mock rankings
        self.ml_input_row['home_current_rank'] = 5
        self.ml_input_row['away_current_rank'] = 8
        self.ml_input_row['home_rank_change'] = -1  # Javult
        self.ml_input_row['away_rank_change'] = 2   # Romlott
        
        logger.info(f"  Home rank: #{self.ml_input_row['home_current_rank']} (change: {self.ml_input_row['home_rank_change']:+d})")
        logger.info(f"  Away rank: #{self.ml_input_row['away_current_rank']} (change: {self.ml_input_row['away_rank_change']:+d})")
    
    def _step9_build_ml_input(self):
        """9️⃣ Build final ML input row."""
        logger.info("\n" + "="*60)
        logger.info("9️⃣ ML INPUT ROW ÉPÍTÉSE")
        logger.info("="*60)
        
        # Basic match info
        self.ml_input_row['match_id'] = self.selected_match['match_id']
        self.ml_input_row['event_id'] = self.event_id
        self.ml_input_row['date'] = self.selected_match['date']
        
        # Date features
        # date_parsed = pd.to_datetime(self.selected_match['date'], errors='coerce')
        # if pd.notna(date_parsed):
        #     self.ml_input_row['date_month'] = date_parsed.month
        #     self.ml_input_row['date_day'] = date_parsed.dayofweek
        # Mock for now
        self.ml_input_row['date_month'] = 9
        self.ml_input_row['date_day'] = 5  # Friday
        
        # Teams
        self.ml_input_row['team_home'] = self.teams['home']['team_name']
        self.ml_input_row['team_away'] = self.teams['away']['team_name']
        
        # H2H features
        self.ml_input_row['H2H_winrate_team1'] = self.match_h2h['home_win_rate']
        self.ml_input_row['H2H_games'] = self.match_h2h['wins_home'] + self.match_h2h['wins_away']
        
        # Match info
        self.ml_input_row['match_rounds'] = self.selected_match['rounds']
        self.ml_input_row['map_veto_decider'] = 1 if self.selected_match['rounds'] >= 3 else 0  # Simplified
        
        # Player stats
        self.ml_input_row['avg_rating_top3_team1'] = self.match_h2h['home_team_avg_rating']  # Simplified (not top3)
        
        # Historical H2H features (last N matches)
        if 'home' in self.historical_h2h:
            self.ml_input_row['home_hist_avg_rating'] = self.historical_h2h['home'].get('avg_rating')
            self.ml_input_row['home_hist_avg_adr'] = self.historical_h2h['home'].get('avg_adr')
            self.ml_input_row['home_hist_avg_swing'] = self.historical_h2h['home'].get('avg_swing')

        if 'away' in self.historical_h2h:
            self.ml_input_row['away_hist_avg_rating'] = self.historical_h2h['away'].get('avg_rating')
            self.ml_input_row['away_hist_avg_adr'] = self.historical_h2h['away'].get('avg_adr')
            self.ml_input_row['away_hist_avg_swing'] = self.historical_h2h['away'].get('avg_swing')

        # Historical difference features
        if (self.ml_input_row.get('home_hist_avg_rating') and 
            self.ml_input_row.get('away_hist_avg_rating')):
            self.ml_input_row['hist_diff_rating'] = (
                self.ml_input_row['home_hist_avg_rating'] - 
                self.ml_input_row['away_hist_avg_rating']
            )

        if (self.ml_input_row.get('home_hist_avg_adr') and 
            self.ml_input_row.get('away_hist_avg_adr')):
            self.ml_input_row['hist_diff_adr'] = (
                self.ml_input_row['home_hist_avg_adr'] - 
                self.ml_input_row['away_hist_avg_adr']
            )

        # Label (score)
        self.ml_input_row['score_home'] = self.selected_match['score_home']
        self.ml_input_row['score_away'] = self.selected_match['score_away']
        self.ml_input_row['label_home_win'] = 1 if self.selected_match['score_home'] > self.selected_match['score_away'] else 0
        
        logger.info("✅ Feature-ök összegyűjtve!")
        
        # Display
        logger.info("\n" + "="*60)
        logger.info("📋 FINAL ML INPUT ROW:")
        logger.info("="*60)
        
        for key, value in self.ml_input_row.items():
            if isinstance(value, float):
                logger.info(f"  {key:30s} = {value:.4f}")
            else:
                logger.info(f"  {key:30s} = {value}")


def main():
    """Main entry point."""
    
    # ========================================
    # KONFIGURÁCIÓ
    # ========================================
    
    TEST_EVENT_ID = "8292"  # ESL Pro League Season 20
    TEST_MATCH_INDEX = 0    # Első meccs
    
    # ========================================
    # RUN
    # ========================================
    
    generator = MLInputGenerator()
    
    try:
        ml_input_row = generator.run(TEST_EVENT_ID, TEST_MATCH_INDEX)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 SIKER! ML INPUT ROW GENERÁLVA")
        logger.info("="*80)
        
        # DataFrame-ként megjelenítés
        ml_df = pd.DataFrame([ml_input_row])
        
        logger.info("\n📊 DataFrame nézet:")
        print("")
        for col in ml_df.columns:
            print(f"{col:30s} = {ml_df[col].iloc[0]}")
        
        # Export (opcionális)
        # ml_df.to_csv('test_ml_input_row.csv', index=False)
        # logger.info("\n💾 Mentve: test_ml_input_row.csv")
        
    except Exception as e:
        logger.error(f"\n❌ HIBA: {e}", exc_info=True)


if __name__ == "__main__":
    main()