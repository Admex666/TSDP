"""
Entry point: event URL-ek beolvasása és pipeline futtatása.
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

def read_event_urls(file_path: str = "event_links.txt"):
    """
    Event URL-ek beolvasása fájlból.
    
    Args:
        file_path: Fájl elérési útja
    
    Returns:
        List of URLs
    """
    if not os.path.exists(file_path):
        logging.error(f"❌ Nincs ilyen fájl: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logging.info(f"📄 {len(urls)} event URL beolvasva: {file_path}")
    return urls


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🚀 CS:GO SCRAPER PROJEKT INDÍTÁSA")
    logger.info("="*60)
    
    # Event URL-ek beolvasása
    event_urls = read_event_urls("event_links.txt")
    
    if not event_urls:
        logger.error("❌ Nincsenek event URL-ek!")
        return
    
    # Pipeline futtatása
    try:
        pipeline = ScrapePipeline()
        pipeline.run(event_urls)
        logger.info("\n✅ SIKERES FUTTATÁS!")
    except Exception as e:
        logger.error(f"\n❌ HIBA A PIPELINE FUTTATÁSAKOR: {e}", exc_info=True)
    

if __name__ == "__main__":
    main()