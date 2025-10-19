"""
CSV fájlok kezelése: append, update, deduplikáció.
"""

import pandas as pd
import os
from typing import List
import logging

logger = logging.getLogger(__name__)


class CSVManager:
    """CSV fájlok inkrementális frissítése."""
    
    @staticmethod
    def append_or_update(csv_path: str, new_df: pd.DataFrame, key_columns: List[str]):
        """
        Hozzáad vagy frissít rekordokat egy CSV-ben.
        
        Args:
            csv_path: CSV fájl elérési útja
            new_df: Új adatok DataFrame
            key_columns: Kulcs oszlopok a deduplikációhoz (pl. ['match_id'])
        """
        if new_df.empty:
            logger.warning(f"Üres DataFrame, nem történik frissítés: {csv_path}")
            return
        
        try:
            # Ha létezik a CSV, betöltjük
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                logger.info(f"Meglévő adatok betöltve: {len(existing_df)} sor ({csv_path})")
                
                # Deduplikáció: új adatok prioritása
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=key_columns, keep='last')
                
                logger.info(f"Deduplikáció után: {len(combined_df)} sor")
            else:
                combined_df = new_df
                logger.info(f"Új CSV létrehozva: {csv_path}")
            
            # Mentés
            combined_df.to_csv(csv_path, index=False)
            logger.info(f"✅ CSV mentve: {csv_path} ({len(combined_df)} sor)")
            
        except Exception as e:
            logger.error(f"❌ Hiba a CSV mentésekor: {csv_path} - {e}")
            raise
    
    @staticmethod
    def ensure_directory(csv_path: str):
        """Biztosítja, hogy a CSV könyvtára létezik."""
        directory = os.path.dirname(csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Könyvtár létrehozva: {directory}")