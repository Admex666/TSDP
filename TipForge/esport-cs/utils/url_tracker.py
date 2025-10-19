"""
Már feldolgozott URL-ek nyilvántartása (duplikáció elkerülése).
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class URLTracker:
    """URL-ek tracking-je (mikor lett utoljára scrape-elve)."""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Betölti a JSON fájlt, ha létezik."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Hiba a JSON betöltésekor: {e}")
                return {"events": {}, "matches": {}, "teams": {}}
        else:
            return {"events": {}, "matches": {}, "teams": {}}
    
    def _save(self):
        """Menti a JSON fájlt."""
        try:
            # Biztosítjuk a könyvtár létezését
            directory = os.path.dirname(self.json_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.debug(f"URL tracker mentve: {self.json_path}")
        except Exception as e:
            logger.error(f"Hiba a JSON mentésekor: {e}")
    
    def is_scraped(self, category: str, identifier: str, max_age_days: int = 7) -> bool:
        """
        Ellenőrzi, hogy egy URL/ID már scrape-elve lett-e a közelmúltban.
        
        Args:
            category: 'events', 'matches', 'teams'
            identifier: pl. event_id, match_id, team_id
            max_age_days: hány napon belül számít frissnek
        
        Returns:
            True, ha már scrape-elve és friss
        """
        if category not in self.data:
            self.data[category] = {}
        
        if identifier in self.data[category]:
            last_scraped = datetime.fromisoformat(self.data[category][identifier])
            age = datetime.now() - last_scraped
            
            if age.days < max_age_days:
                logger.info(f"⏭️  Skip ({category}/{identifier}) - {age.days} napja scrape-elve")
                return True
        
        return False
    
    def mark_scraped(self, category: str, identifier: str):
        """Megjelöl egy URL-t scrape-eltként."""
        if category not in self.data:
            self.data[category] = {}
        
        self.data[category][identifier] = datetime.now().isoformat()
        self._save()
        logger.debug(f"✅ Megjelölve: {category}/{identifier}")