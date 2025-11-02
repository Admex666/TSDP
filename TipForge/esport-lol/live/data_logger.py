"""
Data Logger - Snapshots mentése CSV formátumban
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnapshotLogger:
    """Menti a scrape snapshotokat CSV fájlba"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_csv_path(self, match_id: str) -> str:
        """Generate CSV path for a match"""
        return os.path.join(self.data_dir, f"match_{match_id}_snapshots.csv")
    
    def save_snapshot(self, match_stats: Dict, odds_data: Dict, 
                     predicted_probs: tuple, home_is_blue: bool,
                     match_id: Optional[str] = None) -> bool:
        """
        Save a single snapshot to CSV
        
        Args:
            match_stats: From scrape_match_stats()
            odds_data: From scrape_odds()
            predicted_probs: (prob_blue, prob_red) tuple
            home_is_blue: Team mapping setting
            match_id: Unique match identifier (default: timestamp-based)
        
        Returns:
            True if saved successfully
        """
        try:
            # Generate match_id if not provided
            if match_id is None:
                match_id = datetime.now().strftime("%Y%m%d_%H%M")
            
            csv_path = self.get_csv_path(match_id)
            
            # Flatten data for CSV row
            snapshot = {
                # Timestamps
                'scrape_time': datetime.now().isoformat(),
                'match_timestamp': match_stats['timestamp'],
                'game_time': match_stats['game_time'],
                'game_index': match_stats.get('game_index', 1),
                
                # Team mapping
                'home_is_blue': home_is_blue,
                
                # Blue team stats
                'blue_kills': match_stats['blue_team']['kills'],
                'blue_towers': match_stats['blue_team']['towers'],
                'blue_inhibitors': match_stats['blue_team']['inhibitors'],
                'blue_barons': match_stats['blue_team']['barons'],
                'blue_gold': match_stats['blue_team']['gold'],
                'blue_dragons': len(match_stats['blue_team']['dragons']),
                'blue_dragon_types': json.dumps(match_stats['blue_team']['dragons']),
                
                # Red team stats
                'red_kills': match_stats['red_team']['kills'],
                'red_towers': match_stats['red_team']['towers'],
                'red_inhibitors': match_stats['red_team']['inhibitors'],
                'red_barons': match_stats['red_team']['barons'],
                'red_gold': match_stats['red_team']['gold'],
                'red_dragons': len(match_stats['red_team']['dragons']),
                'red_dragon_types': json.dumps(match_stats['red_team']['dragons']),
                
                # Predictions
                'pred_blue': predicted_probs[0],
                'pred_red': predicted_probs[1],
                
                # Odds (store all markets as JSON)
                'odds_data': json.dumps(odds_data['markets']),
                
                # Derived metrics
                'gold_diff': match_stats['blue_team']['gold'] - match_stats['red_team']['gold'],
                'kill_diff': match_stats['blue_team']['kills'] - match_stats['red_team']['kills'],
                'tower_diff': match_stats['blue_team']['towers'] - match_stats['red_team']['towers'],
            }
            
            # Player stats aggregation
            if 'players' in match_stats and match_stats['players']:
                blue_cs = sum(p['cs'] for p in match_stats['players'][:5])
                red_cs = sum(p['cs'] for p in match_stats['players'][5:])
                snapshot['blue_cs_total'] = blue_cs
                snapshot['red_cs_total'] = red_cs
                snapshot['cs_diff'] = blue_cs - red_cs
            
            # Create dataframe
            df_new = pd.DataFrame([snapshot])
            
            # Append to existing or create new
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
                logger.info(f"📝 Snapshot appended to {csv_path} (Total: {len(df_combined)} snapshots)")
            else:
                df_new.to_csv(csv_path, index=False)
                logger.info(f"📝 New snapshot file created: {csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save snapshot: {e}")
            return False
    
    def load_snapshots(self, match_id: str) -> Optional[pd.DataFrame]:
        """Load all snapshots for a match"""
        csv_path = self.get_csv_path(match_id)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None
    
    def get_latest_snapshot(self, match_id: str) -> Optional[Dict]:
        """Get the most recent snapshot"""
        df = self.load_snapshots(match_id)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None