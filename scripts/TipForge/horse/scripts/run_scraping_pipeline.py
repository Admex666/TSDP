import os
import sys

# Ensure current directory is in path so scripts can be imported cleanly
sys.path.append(os.getcwd())

from scripts.crawl_historical import crawl_historical
from scripts.collect_lovi_odds_bulk import collect_all_odds
from scripts.batch_fetch_historical import batch_fetch_historical

def main():
    print("==========================================================")
    print("         HISTORICAL DATA COLLECTION PIPELINE")
    print("==========================================================")
    
    # Step 1: Scrape Kincsem Park race results
    print("\n[STEP 1/3] Crawling Kincsem Park race cards (2020-2025)...")
    crawl_historical(["2020", "2021", "2022", "2023", "2024", "2025"], "data/historical_results_combined.json")
    
    # Step 2: Scrape BetLovi closing market odds
    print("\n[STEP 2/3] Scraping BetLovi closing odds...")
    collect_all_odds()
    
    # Step 3: Fetch career details for participants
    print("\n[STEP 3/3] Fetching career data for participants...")
    batch_fetch_historical("data/historical_results_combined.json")
    
    print("\n==========================================================")
    print("      DATA COLLECTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == "__main__":
    main()
