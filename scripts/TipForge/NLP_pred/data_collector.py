import soccerdata as sd
import pandas as pd
import os
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RisingBallerDataCollector:
    def __init__(self, league="ENG-Premier League", season="2324", base_path="./data"):
        self.league = league
        self.season = season
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize soccerdata FBref reader
        self.fbref = sd.FBref(leagues=league, seasons=season)
        
        # Stat types we want to collect
        self.stat_types = ['summary', 'passing', 'defense', 'possession', 'misc']
        
    def get_match_ids(self):
        """Fetches all match IDs for the season."""
        logger.info(f"Fetching schedule for {self.league} {self.season}")
        schedule = self.fbref.read_schedule()
        if 'game_id' in schedule.columns:
            return schedule['game_id'].dropna().unique().tolist()
        return []

    def collect_match_data(self, match_id):
        """Collects all player stats for a single match across all stat types."""
        match_dfs = []
        
        for stype in self.stat_types:
            try:
                # logger.info(f"Fetching {stype} stats for match {match_id}")
                df = self.fbref.read_player_match_stats(stat_type=stype, match_id=match_id)
                
                # Flatten multi-index columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
                
                # Reset index to make league/season/game/team/player available as columns
                df = df.reset_index()
                
                # Keep only unique columns to avoid duplication when merging
                # We'll merge on ['league', 'season', 'game', 'team', 'player']
                match_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error fetching {stype} for {match_id}: {e}")
                
        if not match_dfs:
            return None
        
        # Merge all stat types for this match
        from functools import reduce
        final_df = reduce(lambda left, right: pd.merge(
            left, right, 
            on=['league', 'season', 'game', 'team', 'player'], 
            how='outer', 
            suffixes=('', '_dup')
        ), match_dfs)
        
        # Remove duplicate columns from merge
        final_df = final_df.loc[:, ~final_df.columns.str.endswith('_dup')]
        
        return final_df

    def run(self, limit=None):
        """Main loop to collect data for all matches."""
        match_ids = self.get_match_ids()
        if limit:
            match_ids = match_ids[:limit]
            
        logger.info(f"Found {len(match_ids)} matches to process.")
        
        for i, m_id in enumerate(match_ids):
            out_file = self.base_path / f"{m_id}.parquet"
            
            if out_file.exists():
                logger.info(f"[{i+1}/{len(match_ids)}] Match {m_id} already exists. Skipping.")
                continue
                
            logger.info(f"[{i+1}/{len(match_ids)}] Processing match {m_id}...")
            
            try:
                df = self.collect_match_data(m_id)
                if df is not None:
                    df.to_parquet(out_file)
                    # logger.info(f"Saved {m_id}")
                else:
                    logger.warning(f"No data returned for match {m_id}")
            except Exception as e:
                logger.error(f"Failed to process {m_id}: {e}")
            
            # Optional sleep to be kind to FBref
            # time.sleep(1)

        logger.info("Data collection complete.")

if __name__ == "__main__":
    # Example: Run for just 2 matches to test
    collector = RisingBallerDataCollector(league="ENG-Premier League", season="2324")
    collector.run(limit=None)
