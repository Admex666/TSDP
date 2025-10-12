"""
Team history scraper (winrate, streak, stb.).
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)


class TeamHistoryScraper(BaseScraper):
    """Csapat történeti adatainak scrape-elése."""
    
    def scrape_team_history(self, team_id: str) -> pd.DataFrame:
        """
        Csapat történeti összegzésének scrape-elése.
        
        Args:
            team_id: HLTV team ID (pl. "5378")
        
        Returns:
            DataFrame: team_id, scrape_date, last_30d_winrate, last_90d_winrate, 
                      matches_last_7d, days_since_last_match, current_streak, total_matches_found
        """
        
        def _scrape():
            url = f"https://www.hltv.org/results?team={team_id}"
            self._init_driver()
            self.driver.get(url)
            
            logger.info(f"📈 Team history scrape: team_id={team_id}")
            
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-holder")))
            
            time.sleep(2)
            today = pd.Timestamp.now().normalize()
            
            matches = []
            
            try:
                results_holder = self.driver.find_element(By.CLASS_NAME, "results-holder")
                sublists = results_holder.find_elements(By.CLASS_NAME, "results-sublist")
                
                logger.debug(f"  🔍 Talált results-sublist blokkok: {len(sublists)}")
                
                for sublist in sublists:
                    # --- dátum fejléc ---
                    headline_el = sublist.find_element(By.CLASS_NAME, "standard-headline")
                    headline_text = headline_el.text.strip().replace("Results for", "").strip()
                    date = pd.to_datetime(headline_text, errors='coerce')
                    
                    match_blocks = sublist.find_elements(By.CLASS_NAME, "result-con")
                    for match in match_blocks:
                        try:
                            a_tag = match.find_element(By.TAG_NAME, "a")
                            match_url = a_tag.get_attribute("href")
                            table = a_tag.find_element(By.TAG_NAME, "table")
                            tds = table.find_elements(By.TAG_NAME, "td")
                            
                            team1_name = tds[0].find_element(By.CLASS_NAME, "team").text.strip()
                            team2_name = tds[2].find_element(By.CLASS_NAME, "team").text.strip()
                            
                            # --- score ---
                            score_spans = tds[1].find_elements(By.TAG_NAME, "span")
                            score1 = int(score_spans[0].text.strip())
                            score2 = int(score_spans[1].text.strip())
                            
                            # --- winner ---
                            team1_html = tds[0].get_attribute("innerHTML")
                            team2_html = tds[2].get_attribute("innerHTML")
                            
                            if "team-won" in team1_html:
                                winner = team1_name
                            elif "team-won" in team2_html:
                                winner = team2_name
                            else:
                                winner = None
                            
                            # --- target team result ---
                            # Egyszerűsítés: feltételezzük, hogy a team_id a query paraméter alapján az érintett csapat
                            # Ezért team1_name az érintett csapat (de ez nem mindig igaz!)
                            # Finomhangolás: ellenőrizd, melyik csapat ID-ja egyezik
                            result = "win" if score1 > score2 else "loss"
                            
                            matches.append({
                                "date": date,
                                "team1": team1_name,
                                "team2": team2_name,
                                "score1": score1,
                                "score2": score2,
                                "winner": winner,
                                "result": result
                            })
                            
                        except Exception as e:
                            logger.debug(f"    ⚠️ Hiba meccs feldolgozásnál: {e}")
                            continue
                
                # --- DataFrame feldolgozás ---
                df = pd.DataFrame(matches)
                if df.empty:
                    logger.warning(f"  ⚠️ Nincs adat a team_id={team_id} számára!")
                    return pd.DataFrame([{
                        "team_id": team_id,
                        "scrape_date": today,
                        "last_30d_winrate": None,
                        "last_90d_winrate": None,
                        "matches_last_7d": 0,
                        "days_since_last_match": None,
                        "current_streak": 0,
                        "total_matches_found": 0,
                        "source_url": url
                    }])
                
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
                df["days_ago"] = (today - df["date"]).dt.days
                
                # --- winrate számítás ---
                def winrate(df):
                    if len(df) == 0:
                        return None
                    return round((df["result"] == "win").sum() / len(df), 4)
                
                last_30d = df[df["days_ago"] <= 30]
                last_90d = df[df["days_ago"] <= 90]
                last_7d = df[df["days_ago"] <= 7]
                
                last_30d_winrate = winrate(last_30d)
                last_90d_winrate = winrate(last_90d)
                matches_last_7d = len(last_7d)
                
                last_match_date = df["date"].max()
                days_since_last_match = int((today - last_match_date).days) if pd.notna(last_match_date) else None
                
                # --- current streak ---
                current_streak = 0
                if len(df) > 0:
                    last_results = df["result"].tolist()
                    last_outcome = last_results[0]
                    for r in last_results:
                        if r == last_outcome:
                            current_streak += 1
                        else:
                            break
                    if last_outcome == "loss":
                        current_streak *= -1  # negatív streak vesztség esetén
                
                df_summary = pd.DataFrame([{
                    "team_id": team_id,
                    "scrape_date": today,
                    "last_30d_winrate": last_30d_winrate,
                    "last_90d_winrate": last_90d_winrate,
                    "matches_last_7d": matches_last_7d,
                    "days_since_last_match": days_since_last_match,
                    "current_streak": current_streak,
                    "total_matches_found": len(df),
                    "source_url": url
                }])
                
                logger.info(f"✅ Összesen {len(df)} meccs feldolgozva a csapatnál")
                return df_summary
                
            except Exception as e:
                logger.error(f"❌ Hiba a team history scraperben: {e}")
                return pd.DataFrame()
        
        result = self._retry_scrape(_scrape)
        self.close()
        return result if result is not None else pd.DataFrame()