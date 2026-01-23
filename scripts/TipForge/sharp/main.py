import logging
import pandas as pd
from pinnacle_scraper import PinnacleScraper
from tippmix_scraper import TippmixScraper
from engine import ValueBetEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Example URLs for eSports
PINNACLE_URLS = [
    "https://www.pinnacle.com/en/esports/matchups/highlights/",
    "https://www.pinnacle.com/en/esports/games/league-of-legends/matchups/",
    "https://www.pinnacle.com/en/esports/games/cs2/matchups/",
    "https://www.pinnacle.com/en/esports/games/call-of-duty/matchups/",
    "https://www.pinnacle.com/en/esports/games/valorant/matchups/",
    "https://www.pinnacle.com/en/tennis/matchups/"
]
TIPPMIX_URLS = [
    "https://www.tippmixpro.hu/hu/fogadas/i/e-sport/esports/96/counter-strike/186/esport",
    "https://www.tippmixpro.hu/hu/fogadas/i/e-sport/esports/96/league-of-legends-lol/100/esport",
    "https://www.tippmixpro.hu/hu/fogadas/i/e-sport/esports/96/valorant/134/esport",
    "https://www.tippmixpro.hu/hu/fogadas/i/fogadas/tenisz/3/osszes/0/kategoria"
]

MIN_EV = 0.01  # Minimum 1% EV to display
MATCH_THRESHOLD = 80 # Fuzzy matching threshold

def main():
    logger.info("Starting Sharp Value Bet Finder...")
    
    # 1. Scrape Pinnacle (Sharp Odds)
    p_scraper = PinnacleScraper(headless=True)
    p_matches = []
    
    for url in PINNACLE_URLS:
        logger.info(f"Scraping Pinnacle: {url}")
        matches = p_scraper.scrape_matches(url)
        p_matches.extend(matches)
        
    logger.info(f"Retrieved {len(p_matches)} matches from Pinnacle (Total)")
    
    if not p_matches:
        logger.error("No matches found on Pinnacle. Exiting.")
        return

    # 2. Scrape Tippmix (Recreational Odds)
    t_scraper = TippmixScraper(headless=True)
    t_matches = []
    
    for url in TIPPMIX_URLS:
        logger.info(f"Scraping Tippmix: {url}")
        matches = t_scraper.scrape_matches(url)
        t_matches.extend(matches)
        
    logger.info(f"Retrieved {len(t_matches)} matches from Tippmix (Total)")
    
    if not t_matches:
        logger.error("No matches found on Tippmix. Exiting.")
        return

    # 3. Process and Find Value
    engine = ValueBetEngine(min_ev=MIN_EV, match_threshold=MATCH_THRESHOLD)
    value_bets = engine.find_value_bets(p_matches, t_matches)
    
    # 4. Display Results
    if value_bets:
        df = pd.DataFrame(value_bets)
        # Format for better display
        df['ev'] = (df['ev'] * 100).round(2).astype(str) + '%'
        df['tippmix_odds'] = df['tippmix_odds'].round(3)
        df['pinnacle_no_vig'] = df['pinnacle_no_vig'].round(3)
        df['fair_prob'] = (df['fair_prob'] * 100).round(1).astype(str) + '%'
        
        print("\n" + "="*80)
        print(f" FOUND {len(value_bets)} VALUE BETS (Min EV: {MIN_EV*100}%)")
        print("="*80)
        print(df[['match', 'outcome', 'tippmix_odds', 'pinnacle_no_vig', 'ev', 'fair_prob']].to_string(index=False))
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print(" No value bets found at this time.")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
