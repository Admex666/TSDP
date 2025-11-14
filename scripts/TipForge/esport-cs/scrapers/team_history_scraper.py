"""
Team history scraper (winrate, streak, stb.).
"""

from .base_scraper import BaseScraper
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

class TeamHistoryScraper(BaseScraper):
    def scrape_team_matches(self, team_id: str, max_matches: int = 20):
        """Team match history scraping"""
        url = f"https://www.hltv.org/results?team={team_id}"
        self._init_driver()
        self.driver.get(url)

        wait = WebDriverWait(self.driver, 20)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-holder")))

        self._random_delay()

        matches = []
        all_sublists = self.driver.find_elements(By.CLASS_NAME, "results-sublist")
        print(f"🔍 Összesen {len(all_sublists)} results-sublist betöltve")

        for sublist in all_sublists:
            if len(matches) >= max_matches:
                break

            try:
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
                    print(f"⚠️ Hiba egy meccs feldolgozásánál: {e}")
                    continue

        self.close()
        return pd.DataFrame(matches)