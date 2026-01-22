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

def main():
    # --- CONFIGURATION ---
    # Example URLs for Premier League (Soccer)
    PINNACLE_URL = "https://www.pinnacle.com/en/soccer/england-premier-league/matchups/"
    TIPPMIX_URL = "https://www.tippmixpro.hu/hu/fogadas/i/bajnoksag-lokacio/labdarugas/1/anglia/77/premier-liga/272212442942148608"
    
    MIN_EV = 0.01  # Minimum 1% EV to display
    MATCH_THRESHOLD = 80 # Fuzzy matching threshold
    
    logger.info("Starting Sharp Value Bet Finder...")
    
    # 1. Scrape Pinnacle (Sharp Odds)
    p_scraper = PinnacleScraper(headless=True)
    logger.info(f"Scraping Pinnacle: {PINNACLE_URL}")
    p_matches = p_scraper.scrape_matches(PINNACLE_URL)
    logger.info(f"Retrieved {len(p_matches)} matches from Pinnacle")
    
    if not p_matches:
        logger.error("No matches found on Pinnacle. Exiting.")
        return

    # 2. Scrape Tippmix (Recreational Odds)
    t_scraper = TippmixScraper(headless=True)
    logger.info(f"Scraping Tippmix: {TIPPMIX_URL}")
    t_matches = t_scraper.scrape_matches(TIPPMIX_URL)
    logger.info(f"Retrieved {len(t_matches)} matches from Tippmix")
    
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
