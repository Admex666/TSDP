import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class TippmixDiscovery:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.base_url = "https://www.tippmixpro.hu/hu/fogadas/esport"

    def _setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        return webdriver.Chrome(options=chrome_options)

    def discover_lol_matches(self) -> Dict[str, str]:
        """
        Discover live and upcoming LoL match URLs on Tippmix.
        Returns mapping of "team1_vs_team2" -> url
        """
        driver = None
        mappings = {}
        try:
            # Use specific LoL listing URL provided by user investigation
            target_url = "https://www.tippmixpro.hu/hu/fogadas/i/fogadas/league-of-legends-lol/100/osszes/0/helyszin"
            logger.info(f"🌐 Navigating to Tippmix LoL: {target_url}")
            driver = self._setup_driver()
            driver.get(target_url)
            
            # Accept cookies
            try:
                cookie_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                )
                cookie_btn.click()
                time.sleep(1)
            except:
                pass
            
            # Switch to sports content iframe
            logger.info("⏳ Waiting for Sports Iframe (src*='sports2')...")
            try:
                iframe = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='sports2']"))
                )
                driver.switch_to.frame(iframe)
                time.sleep(5) # Allow content to populate
            except Exception as e:
                logger.error(f"Could not find or switch to sports iframe: {e}")
                return {}
            
            # --- Try enabling LIVE matches ---
            try:
                logger.info("🖱️ Searching for 'ÉLŐ' toggle to enable live matches...")
                # Search by text content
                live_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'ÉLŐ') or contains(text(), 'Élő') or contains(text(), 'LIVE')]")
                for el in live_elements:
                    # Filter out script/style/irrelevant
                    if el.tag_name in ['script', 'style', 'noscript']: continue
                    
                    # Check text length to avoid clicking paragraphs
                    text = el.text.strip()
                    if text and len(text) < 15 and ("ÉLŐ" in text.upper() or "LIVE" in text.upper()):
                        logger.info(f"👉 Clicking potential Live button: '{text}' ({el.get_attribute('class')})")
                        el.click()
                        time.sleep(3) # Wait for live content load
                        break
            except Exception as live_e:
                logger.warning(f"Could not click LIVE button: {live_e}")
            # ----------------------------------

            # Extract matches
            # We look for all links inside the iframe that point to an event
            links = driver.find_elements(By.TAG_NAME, "a")
            logger.info(f"🔍 Found {len(links)} total links in iframe. Filtering for matches...")
            
            for link in links:
                try:
                    url = link.get_attribute("href")
                    # Filter for match detail pages
                    if not url or ("fogadas/i/esemenyek" not in url and "fogadas/esemenyek" not in url):
                        continue
                     
                    if "league-of-legends-lol" not in url:
                        continue

                    # Text usually contains "Team1\nTeam2\nDate\nTime"
                    text = link.text.strip()
                    parts = [p.strip() for p in text.split("\n") if p.strip()]
                    
                    # Heuristic: Teams are usually the first two non-numeric, non-date parts
                    # Or we can just use the URL slug which is safer: /team1-team2/
                    
                    key = None
                    
                    # Strategy A: Text based (Preferred if available)
                    if len(parts) >= 2:
                        t1 = parts[0]
                        t2 = parts[1]
                        # Verify they look like names and not "01.14" etc
                        if not t1[0].isdigit() and not t2[0].isdigit():
                            key = f"{t1}_vs_{t2}".replace(" ", "_")
                    
                    # Strategy B: URL Slug Fallback
                    if not key:
                        # expected format: .../vilag/tournament/team1-team2/id
                        # split by / and take the 2nd to last part usually
                        segments = url.split('/')
                        # Filter out empty strings and find the segment with team names
                        # Usually it is before the numeric ID at the end
                        if segments[-1].isdigit():
                            slug = segments[-2]
                            key = slug.replace("-", "_vs_")
                        else:
                            # Try finding the segment with dashes
                            for s in reversed(segments):
                                if "-" in s and not s[0].isdigit():
                                    key = s.replace("-", "_vs_")
                                    break
                    
                    if key and key not in mappings:
                        mappings[key] = url
                        logger.info(f"  -> Found Match: {key} ({url})")
                            
                except Exception as loop_e:
                    # logger.debug(f"Error parsing link: {loop_e}")
                    continue
                    
            logger.info(f"✅ Discovered {len(mappings)} LoL matches on Tippmix.")
            return mappings
            
        except Exception as e:
            logger.error(f"❌ Discovery error: {e}")
            # Take screenshot for debugging if it fails
            if driver:
                try:
                    driver.save_screenshot("discovery_error.png")
                    logger.info("📸 Saved error screenshot to discovery_error.png")
                except: pass
            return {}
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)
    discovery = TippmixDiscovery(headless=True)
    matches = discovery.discover_lol_matches()
    for m, url in matches.items():
        print(f"MATCH: {m} -> {url}")
