"""
Head-to-Head és player stats scraper.
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class H2HScraper(BaseScraper):
    """Match H2H és player stats scrape-elése."""
    
    def scrape_match_h2h(self, match_url: str) -> pd.DataFrame:
        """
        Match H2H és player stats scrape-elése.
        
        Args:
            match_url: Teljes HLTV match URL
        
        Returns:
            DataFrame: match_id, home_team, away_team, wins_home, wins_away, overtimes,
                      home_win_rate, home/away team stats (rating, ADR, swing avg/std)
        """
        
        def _scrape():
            self._init_driver()
            self.driver.get(match_url)
            
            match_id = match_url.split('/')[4]
            logger.info(f"🔍 H2H scrape: match_id={match_id}")
            
            wait = WebDriverWait(self.driver, 15)
            data = []
            
            try:
                # --- HEAD TO HEAD rész ---
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "head-to-head")))
                h2h_box = self.driver.find_element(By.CLASS_NAME, "head-to-head")
                container = h2h_box.find_element(By.CLASS_NAME, "standard-box")
                
                team1 = container.find_element(By.CSS_SELECTOR, ".team1 .teamName").text.strip()
                team2 = container.find_element(By.CSS_SELECTOR, ".team2 .teamName").text.strip()
                
                # Team ID-k kinyerése
                team1_id = None
                team2_id = None

                try:
                    team1_links = container.find_elements(By.CSS_SELECTOR, ".team1 a")
                    for link in team1_links:
                        href = link.get_attribute("href")
                        if href and '/team/' in href:
                            parts = href.split('/')
                            if len(parts) >= 5 and parts[3] == 'team':
                                team1_id = parts[4]
                                break
                except Exception as e:
                    logger.debug(f"  ⚠️ Team1 ID hiba: {e}")

                try:
                    team2_links = container.find_elements(By.CSS_SELECTOR, ".team2 a")
                    for link in team2_links:
                        href = link.get_attribute("href")
                        if href and '/team/' in href:
                            parts = href.split('/')
                            if len(parts) >= 5 and parts[3] == 'team':
                                team2_id = parts[4]
                                break
                except Exception as e:
                    logger.debug(f"  ⚠️ Team2 ID hiba: {e}")

                logger.debug(f"  Team IDs: {team1} = {team1_id}, {team2} = {team2_id}")

                mapholders = self.driver.find_elements(By.CLASS_NAME, "mapholder")

                map_data = []
                for mapholder in mapholders:
                    try:
                        mapname = mapholder.find_element(By.CLASS_NAME, "mapname").text.strip()
                        
                        results_div = mapholder.find_element(By.XPATH, ".//*[contains(@class, 'results ')]")
                        results_class = results_div.get_attribute("class")
                        played = 'played' in results_class

                        if not played:
                            map_data.append({
                                "map_name": mapname,
                                "played": played,
                                "home_win": None,
                                "home_pick": None,
                                "home_score": -1,
                                "away_score": -1
                            })
                            continue

                        # Bal és jobb oldal div-ek dinamikus classokkal
                        left_div = mapholder.find_element(By.XPATH, ".//*[contains(@class, 'results-left')]")
                        right_div = mapholder.find_element(By.XPATH, ".//*[contains(@class, 'results-right')]")

                        # class attribútum alapján állapotok (won / lost / pick)
                        left_classes = left_div.get_attribute("class")

                        left_win = "win" if "won" in left_classes else "loss" if "lost" in left_classes else "unknown"

                        left_pick = "pick" in left_classes

                        # score kiolvasás
                        left_score = left_div.find_element(By.CLASS_NAME, "results-team-score").text
                        right_score = right_div.find_element(By.CLASS_NAME, "results-team-score").text

                        map_data.append({
                            "map_name": mapname,
                            "played": played,
                            "home_win": left_win,
                            "home_pick": left_pick,
                            "home_score": left_score,
                            "away_score": right_score
                        })

                    except Exception as e:
                        print(f"⚠️ Hiba egy map feldolgozásánál: {e}")
                        continue

                import json

                # --- Map-level feature aggregálás ---
                if map_data:
                    maps_played = sum(1 for m in map_data if m["played"])
                    maps_won = sum(1 for m in map_data if m["home_win"] == "win")
                    maps_lost = sum(1 for m in map_data if m["home_win"] == "loss")
                    maps_picked_by_home = sum(1 for m in map_data if m["home_pick"])
                    avg_score_diff = np.mean([
                        int(m["home_score"]) - int(m["away_score"])
                        for m in map_data if m["played"]
                    ]) if maps_played > 0 else None

                    # Map winrate (arány)
                    map_winrate = maps_won / maps_played if maps_played > 0 else None

                    # JSON formában is tároljuk, ha később map-szinten akarjuk kinyerni
                    map_json = json.dumps(map_data, ensure_ascii=False)

                else:
                    maps_played = maps_won = maps_lost = maps_picked_by_home = 0
                    avg_score_diff = map_winrate = None
                    map_json = "[]"

                stats = container.find_elements(By.CSS_SELECTOR, ".flexbox-column.grow .bold")
                wins_team1 = int(stats[0].text.strip())
                overtimes = int(stats[1].text.strip())
                wins_team2 = int(stats[2].text.strip())
                total_non_ot = wins_team1 + wins_team2
                home_win_rate = wins_team1 / total_non_ot if total_non_ot > 0 else None
                
                # --- PLAYER STAT rész ---
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "stats-content")))
                time.sleep(2)  # Extra várakozás a dinamikus tartalom betöltésére
                
                stats_tables = self.driver.find_elements(By.CSS_SELECTOR, "table.totalstats")
                
                def parse_team_stats(table):
                    """Parseolja a team táblát és visszaadja a statok listáját."""
                    rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # első sor a header
                    team_name = table.find_element(By.CSS_SELECTOR, ".teamName.team").text.strip()
                    
                    ratings, adrs, swings = [], [], []
                    
                    for r in rows:
                        try:
                            rating = float(r.find_element(By.CSS_SELECTOR, ".rating").text.strip())
                            adr = float(r.find_element(By.CSS_SELECTOR, ".adr").text.strip())
                            swing_text = r.find_element(By.CSS_SELECTOR, ".roundSwing").text.strip().replace('%','')
                            swing = float(swing_text.replace('+', '').replace(',', '.'))
                            ratings.append(rating)
                            adrs.append(adr)
                            swings.append(swing)
                        except Exception:
                            continue
                    
                    return team_name, ratings, adrs, swings
                
                # Feltételezzük, hogy az első totalstats a home team (team1), a második az away (team2)
                if len(stats_tables) >= 2:
                    home_team_name, home_ratings, home_adrs, home_swings = parse_team_stats(stats_tables[0])
                    away_team_name, away_ratings, away_adrs, away_swings = parse_team_stats(stats_tables[1])
                else:
                    logger.warning(f"⚠️ Nem talált elég player stats táblát")
                    home_ratings = home_adrs = home_swings = []
                    away_ratings = away_adrs = away_swings = []
                
                data.append({
                    "match_id": match_id,
                    "home_team": team1,
                    "home_team_id": team1_id,
                    "away_team": team2,
                    "away_team_id": team2_id,
                    "wins_home": wins_team1,
                    "wins_away": wins_team2,
                    "overtimes": overtimes,
                    "total_non_overtime": total_non_ot,
                    "home_win_rate": round(home_win_rate, 4) if home_win_rate is not None else None,
                    "home_team_avg_rating": round(np.mean(home_ratings), 4) if home_ratings else None,
                    "home_team_std_rating": round(np.std(home_ratings), 4) if home_ratings else None,
                    "home_team_avg_ADR": round(np.mean(home_adrs), 2) if home_adrs else None,
                    "home_team_std_ADR": round(np.std(home_adrs), 2) if home_adrs else None,
                    "home_team_avg_Swing": round(np.mean(home_swings), 2) if home_swings else None,
                    "home_team_std_Swing": round(np.std(home_swings), 2) if home_swings else None,
                    "away_team_avg_rating": round(np.mean(away_ratings), 4) if away_ratings else None,
                    "away_team_std_rating": round(np.std(away_ratings), 4) if away_ratings else None,
                    "away_team_avg_ADR": round(np.mean(away_adrs), 2) if away_adrs else None,
                    "away_team_std_ADR": round(np.std(away_adrs), 2) if away_adrs else None,
                    "away_team_avg_Swing": round(np.mean(away_swings), 2) if away_swings else None,
                    "away_team_std_Swing": round(np.std(away_swings), 2) if away_swings else None,
                    "maps_played": maps_played,
                    "home_maps_won": maps_won,
                    "away_maps_won": maps_lost,
                    "home_maps_picked": maps_picked_by_home,
                    "map_avg_score_diff": avg_score_diff,
                    "home_map_winrate": map_winrate,
                    "maps_json": map_json,
                    "source_url": match_url
                })
                
                logger.info(f"✅ H2H scrape-elve: {team1} vs {team2}")

                # --- TEAMS.CSV MENTÉS ---
                if team1_id or team2_id:
                    teams_data = []
                    if team1_id:
                        teams_data.append({"team_id": team1_id, "team_name": team1})
                    if team2_id:
                        teams_data.append({"team_id": team2_id, "team_name": team2})
                    
                    if teams_data:
                        from utils import CSVManager
                        from utils.config import TEAMS_CSV
                        teams_df = pd.DataFrame(teams_data)
                        csv_manager = CSVManager()
                        csv_manager.append_or_update(TEAMS_CSV, teams_df, ['team_id'])
                        logger.debug(f"  ✅ Teams mentve: {len(teams_data)} csapat")

            except Exception as e:
                logger.error(f"❌ Hiba a H2H scraperben: {e}")
                return pd.DataFrame()

            return pd.DataFrame(data)
    
        result = self._retry_scrape(_scrape)
        self.close()
        return result if result is not None else pd.DataFrame()