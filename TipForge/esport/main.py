"""
Entry point: event archive scraping és pipeline futtatása.
"""

import logging
import os
from pipeline import ScrapePipeline
from utils.config import LOG_DIR, LOG_FILE

# 🔧 LOGGING SETUP
def setup_logging():
    """Logging konfiguráció."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🚀 CS:GO SCRAPER PROJEKT INDÍTÁSA")
    logger.info("="*60)
    
    # Pipeline futtatása az archive-ból
    try:
        pipeline = ScrapePipeline()
        
        pipeline.run_from_archive("https://www.hltv.org/events/archive?team=6667")
        
        logger.info("\n✅ SIKERES FUTTATÁS!")
    except Exception as e:
        logger.error(f"\n❌ HIBA A PIPELINE FUTTATÁSAKOR: {e}", exc_info=True)
    

if __name__ == "__main__":
    main()