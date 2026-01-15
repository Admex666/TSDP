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
        Discover live LoL match URLs on Tippmix.
        Returns mapping of "team1_vs_team2" -> url
        """
        driver = None
        mappings = {}
        try:
            # Use the e-sport page, then click "Élőben" to show live matches
            target_url = "https://www.tippmixpro.hu/hu/fogadas/i/e-sport/esports/96/league-of-legends-lol/100/esport"
            logger.info(f"🌐 Navigating to Tippmix E-sport LoL: {target_url}")
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
            
            # Click "Élőben" button to show LIVE matches
            logger.info("🖱️ Searching for 'Élőben' button to show live matches...")
            try:
                # Search by text content for "Élőben" or "Élő"
                live_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Élőben') or contains(text(), 'Élő')]")
                clicked = False
                for el in live_elements:
                    # Filter out script/style/irrelevant
                    if el.tag_name in ['script', 'style', 'noscript']: 
                        continue
                    
                    # Check text length to avoid clicking paragraphs
                    text = el.text.strip()
                    if text and len(text) < 15 and ("ÉLŐBEN" in text.upper() or "ÉLŐ" == text.upper()):
                        logger.info(f"👉 Found Live button: '{text}' (class: {el.get_attribute('class')})")
                        
                        # Try clicking the parent element (the actual clickable span)
                        try:
                            parent = el.find_element(By.XPATH, "..")
                            logger.info(f"   Parent class: {parent.get_attribute('class')}")
                            # Use JavaScript to click to avoid interception
                            driver.execute_script("arguments[0].click();", parent)
                            logger.info("   ✓ Clicked via JavaScript")
                            time.sleep(5) # Wait for live content to load
                            clicked = True
                            break
                        except:
                            # Fallback: try clicking the element itself with JS
                            try:
                                driver.execute_script("arguments[0].click();", el)
                                logger.info("   ✓ Clicked element via JavaScript")
                                time.sleep(5)
                                clicked = True
                                break
                            except:
                                pass
                
                if not clicked:
                    logger.warning("Could not find or click 'Élőben' button")
            except Exception as live_e:
                logger.warning(f"Error clicking LIVE button: {live_e}")

            # Extract matches - ONLY look for elo-esemenyek URLs (live matches)
            links = driver.find_elements(By.TAG_NAME, "a")
            logger.info(f"🔍 Found {len(links)} total links in iframe. Filtering for LIVE matches...")
            
            for link in links:
                try:
                    url = link.get_attribute("href")
                    if not url:
                        continue
                    
                    # CRITICAL: Only accept elo-esemenyek URLs (live matches)
                    if "elo-esemenyek" not in url:
                        continue
                     
                    if "league-of-legends-lol" not in url:
                        continue

                    # Text usually contains "Team1\nTeam2\nScore" for live matches
                    text = link.text.strip()
                    parts = [p.strip() for p in text.split("\n") if p.strip()]
                    
                    key = None
                    
                    # Strategy A: Text based (Preferred if available)
                    if len(parts) >= 2:
                        t1 = parts[0]
                        t2 = parts[1]
                        # Verify they look like names and not "01.14" or scores
                        if not t1[0].isdigit() and not t2[0].isdigit():
                            key = f"{t1}_vs_{t2}".replace(" ", "_")
                    
                    # Strategy B: URL Slug Fallback
                    if not key:
                        # expected format: .../vilag/tournament/team1-team2/id
                        segments = url.split('/')
                        # Usually team names are before the numeric ID at the end
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
                        logger.info(f"  -> Found LIVE Match: {key}")
                            
                except Exception as loop_e:
                    continue
                    
            logger.info(f"✅ Discovered {len(mappings)} LIVE LoL matches on Tippmix.")
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
