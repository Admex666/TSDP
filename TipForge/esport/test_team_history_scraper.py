import time
import random
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==============================
#   BASE SELENIUM SCRAPER CLASS
# ==============================

class BaseScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None

    def _init_driver(self):
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1400,1600")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = webdriver.Chrome(options=options)

    def _random_delay(self, min_delay=1.5, max_delay=3.5):
        time.sleep(random.uniform(min_delay, max_delay))

    def close(self):
        if self.driver:
            self.driver.quit()


# ==========================================
#   SCRAPER – HLTV TEAM MATCH HISTORY
# ==========================================

class TeamHistoryScraper(BaseScraper):
    def scrape(self, team_id: str, max_matches: int = 100):
        url = f"https://www.hltv.org/results?team={team_id}"
        self._init_driver()  # nagy viewport
        self.driver.get(url)

        wait = WebDriverWait(self.driver, 20)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "results-holder")))

        matches = []
        seen_ids = set()

        # scroll loop
        prev_count = 0
        same_counter = 0
        max_same = 3

        while True:
            # jelenlegi blokkok
            all_blocks = self.driver.find_elements(By.CLASS_NAME, "result-con")
            curr_count = len(all_blocks)

            if curr_count == prev_count:
                same_counter += 1
            else:
                same_counter = 0
                prev_count = curr_count

            # ha elértük a maxot vagy nem nő tovább
            if curr_count >= max_matches or same_counter >= max_same:
                break

            # görgetünk lejjebb
            self.driver.execute_script("window.scrollBy(0, 1200);")
            time.sleep(1.2)

        # kis biztonsági késleltetés
        time.sleep(1.5)
        
        all_sublists = self.driver.find_elements(By.CLASS_NAME, "results-sublist")
        print(f"🔍 Összesen {len(all_blocks)} blokk betöltve a DOM-ban.")

        # --- új feldolgozás sublist alapon ---
        results_holder = self.driver.find_element(By.CLASS_NAME, "results-holder")
        sublists = results_holder.find_elements(By.CLASS_NAME, "results-sublist")

        print(f"🔍 {len(sublists)} napnyi results-sublist betöltve a DOM-ban.")

        for sublist in all_sublists:
            try:
                # dátum kiolvasása a headline-ból
                try:
                    headline = sublist.find_element(By.CLASS_NAME, "standard-headline").text.strip()
                    match_date = headline.replace("Results for", "").strip()
                except:
                    match_date = None

                blocks = sublist.find_elements(By.CLASS_NAME, "result-con")
                print(f"📅 {match_date}: {len(blocks)} meccs")

                for i, block in enumerate(blocks):
                    if len(matches) >= max_matches:
                        break

                    try:
                        a = block.find_element(By.TAG_NAME, "a")
                        href = a.get_attribute("href")
                        if not href or "/matches/" not in href:
                            continue

                        match_id_match = re.search(r"/matches/(\d+)/", href)
                        if not match_id_match:
                            continue
                        match_id = match_id_match.group(1)
                        if match_id in seen_ids:
                            continue
                        seen_ids.add(match_id)

                        table = a.find_element(By.TAG_NAME, "table")
                        tds = table.find_elements(By.TAG_NAME, "td")
                        if len(tds) < 3:
                            continue

                        team1_name = tds[0].find_element(By.CLASS_NAME, "team").text.strip()
                        team2_name = tds[2].find_element(By.CLASS_NAME, "team").text.strip()

                        spans = tds[1].find_elements(By.TAG_NAME, "span")
                        score1 = int(spans[0].text) if len(spans) > 1 else None
                        score2 = int(spans[1].text) if len(spans) > 1 else None

                        team1_html = tds[0].get_attribute("innerHTML")
                        team2_html = tds[2].get_attribute("innerHTML")
                        team1_won = "team-won" in team1_html
                        team2_won = "team-won" in team2_html
                        result = "win" if team1_won else "loss" if team2_won else "unknown"

                        map_text = a.find_element(By.CSS_SELECTOR, ".map-text").text.strip() if a.find_elements(By.CSS_SELECTOR, ".map-text") else "bo1"

                        matches.append({
                            "team_id": team_id,
                            "match_id": match_id,
                            "match_date": match_date,
                            "team1": team1_name,
                            "team2": team2_name,
                            "result": result,
                            "score_for": score1,
                            "score_against": score2,
                            "map_type": map_text,
                            "link": href
                        })

                    except Exception as e:
                        print(f"💥 Hiba egy meccsnél: {e}")
                        continue

            except Exception as e:
                print(f"💥 Hiba sublist feldolgozásakor: {e}")
                continue


        self.close()
        df = pd.DataFrame(matches)
        return df




# ==============================
#         TESZT FUTTATÁS
# ==============================

if __name__ == "__main__":
    team_id = 4494  # MOUZ
    scraper = TeamHistoryScraper(headless=False)
    df = scraper.scrape(team_id, max_matches=50)

    print("\n✅ SCRAPE KÉSZ – EREDMÉNYEK:")
    print(df)

    if not df.empty:
        print("\n📊 Első 5 meccs:")
        print(df[["match_date", "team1", "team2", "result", "score_for", "score_against", "map_type"]].head())