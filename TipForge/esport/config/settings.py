"""
Configuration settings for CS:GO scraper
"""
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for directory in [RAW_DIR, PROCESSED_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "cs_scraper.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Scraper settings
HEADLESS = True
RANDOM_DELAY_MIN = 2
RANDOM_DELAY_MAX = 5
PAGE_LOAD_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_DELAY = 5

# Feature settings
N_HISTORICAL_MATCHES = 3  # Number of matches to scrape for historical H2H

# URLs
HLTV_BASE = "https://www.hltv.org"
ODDSPORTAL_BASE = "https://www.oddsportal.com"

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"