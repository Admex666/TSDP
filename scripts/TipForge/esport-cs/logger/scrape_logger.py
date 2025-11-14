"""
Logging system for scraping operations
"""
import logging
from pathlib import Path
from datetime import datetime
from config.settings import LOGS_DIR, LOG_FORMAT, LOG_LEVEL


class ScrapeLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        
        # File handler (daily rotation)
        log_file = LOGS_DIR / f"scraper_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def scrape_start(self, scrape_type: str, identifier: str):
        self.logger.info(f"🔵 START {scrape_type}: {identifier}")
    
    def scrape_success(self, scrape_type: str, identifier: str, count: int = None):
        msg = f"✅ SUCCESS {scrape_type}: {identifier}"
        if count is not None:
            msg += f" ({count} items)"
        self.logger.info(msg)
    
    def scrape_error(self, scrape_type: str, identifier: str, error: str):
        self.logger.error(f"❌ ERROR {scrape_type}: {identifier} - {error}")
    
    def scrape_skip(self, scrape_type: str, identifier: str, reason: str):
        self.logger.info(f"⏭️ SKIP {scrape_type}: {identifier} - {reason}")
    
    def progress(self, current: int, total: int, item: str = "items"):
        self.logger.info(f"📊 Progress: {current}/{total} {item} ({current/total*100:.1f}%)")