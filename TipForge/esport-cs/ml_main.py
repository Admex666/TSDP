"""
ML pipeline entry point.
"""

import logging
import os
from ml_pipeline import MLScrapePipeline, FeatureBuilder
from ml_pipeline.ml_config import ML_DATA_DIR
from utils.config import LOG_DIR, LOG_FILE

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
    """ML pipeline main."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("🚀 ML PIPELINE INDÍTÁSA")
    logger.info("="*80)
    
    # ========================================
    # 1. SCRAPING PHASE
    # ========================================
    
    # Definiáld az event ID-kat, amiket scrape-elni szeretnél
    TARGET_EVENT_IDS = [
        "7441",  # ESL Pro League Season 20
        # "7902",  # BLAST.tv Paris Major 2023
        # Adj hozzá többet...
    ]
    
    logger.info(f"\n🎯 Target events: {TARGET_EVENT_IDS}")
    
    try:
        pipeline = MLScrapePipeline()
        pipeline.run(TARGET_EVENT_IDS)
        logger.info("\n✅ SCRAPING BEFEJEZVE!")
    except Exception as e:
        logger.error(f"\n❌ HIBA A SCRAPING-BAN: {e}", exc_info=True)
        return
    
    # ========================================
    # 2. FEATURE ENGINEERING PHASE
    # ========================================
    
    logger.info(f"\n{'='*80}")
    logger.info("🔧 FEATURE ENGINEERING INDÍTÁSA")
    logger.info(f"{'='*80}")
    
    try:
        feature_builder = FeatureBuilder()
        ml_dataset = feature_builder.build_ml_dataset()
        
        logger.info(f"\n✅ ML DATASET KÉSZ!")
        logger.info(f"📊 Sorok: {len(ml_dataset)}")
        logger.info(f"📊 Oszlopok: {len(ml_dataset.columns)}")
        logger.info(f"📊 Fájl: ml_data/ml_dataset.csv")
        
    except Exception as e:
        logger.error(f"\n❌ HIBA A FEATURE ENGINEERING-BEN: {e}", exc_info=True)
        return
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ML PIPELINE SIKERES BEFEJEZÉS!")
    logger.info("="*80)


if __name__ == "__main__":
    main()