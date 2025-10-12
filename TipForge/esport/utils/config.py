"""
Konfigurációs beállítások a scraper projekthez.
"""

# Selenium beállítások
SCRAPE_DELAY_MIN = 2  # sec
SCRAPE_DELAY_MAX = 5
RETRY_ATTEMPTS = 3
HEADLESS_MODE = False
PAGE_LOAD_TIMEOUT = 15

# URL cache beállítások
URL_CACHE_DAYS = 7  # Hány napon belül ne scrape-elje újra ugyanazt

# CSV fájlok
DATA_DIR = "data"
EVENTS_CSV = f"{DATA_DIR}/events.csv"
RESULTS_CSV = f"{DATA_DIR}/results.csv"
MATCHES_H2H_CSV = f"{DATA_DIR}/matches_h2h.csv"
TEAMS_CSV = f"{DATA_DIR}/teams.csv"
TEAM_HISTORY_CSV = f"{DATA_DIR}/team_history.csv"
RANKINGS_CSV = f"{DATA_DIR}/rankings.csv"
SCRAPED_URLS_JSON = f"{DATA_DIR}/scraped_urls.json"

# Logging
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/scraper.log"