"""
ML pipeline specifikus konfiguráció.
"""

# Könyvtárak
ML_DATA_DIR = "ml_data"
ML_TARGET_MATCHES_CSV = f"{ML_DATA_DIR}/target_matches.csv"
ML_MATCH_DETAILS_CSV = f"{ML_DATA_DIR}/match_details.csv"
ML_TEAM_HISTORY_CSV = f"{ML_DATA_DIR}/team_match_history.csv"
ML_DATASET_CSV = f"{ML_DATA_DIR}/ml_dataset.csv"
ML_SCRAPED_URLS_JSON = f"{ML_DATA_DIR}/ml_scraped_urls.json"

# Feature engineering
ROLLING_WINDOW_SIZES = [3, 5]  # Utolsó N meccs
H2H_LOOKBACK_DAYS = 365  # 1 év H2H történet

# Scraping
ML_URL_CACHE_DAYS = 30  # ML adatok ritkábban változnak