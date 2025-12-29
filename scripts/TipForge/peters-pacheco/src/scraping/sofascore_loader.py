import pandas as pd
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from .sofascore import (
    get_events_for_round,
    create_lineups_df,
    create_statistics_df,
    create_shotmap_df,
    create_average_positions_df,
    create_graph_df,
    create_odds_df,
    scrape_sofascore
)

class SofaScoreLoader:
    def __init__(self, data_dir: str = "data/sofascore", tournament_id: int = 17, season_id: int = 52186):
        """
        Loader for SofaScore data.
        PL 23/24 IDs: Tournament 17, Season 52186.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.events_dir = self.processed_dir / "events"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        
        self.tournament_id = tournament_id
        self.season_id = season_id

    def load_season_schedule(self, force_update: bool = False) -> pd.DataFrame:
        """
        Fetches all events for rounds 1-38 and saves to a consolidated schedule CSV.
        """
        schedule_path = self.processed_dir / f"schedule_{self.tournament_id}_{self.season_id}.csv"
        
        if schedule_path.exists() and not force_update:
            print(f"Loading schedule from {schedule_path}")
            return pd.read_csv(schedule_path)
            
        print("Fetching season schedule from SofaScore...")
        all_events = []
        
        # PL has 38 rounds
        for r in tqdm(range(1, 39), desc="Fetching Rounds"):
            events = get_events_for_round(self.tournament_id, self.season_id, r)
            for e in events:
                # Basic info
                row = {
                    "id": e['id'],
                    "round": r,
                    "startTimestamp": e['startTimestamp'],
                    "status": e['status']['type'],
                    "home_team_id": e['homeTeam']['id'],
                    "home_team": e['homeTeam']['name'],
                    "away_team_id": e['awayTeam']['id'],
                    "away_team": e['awayTeam']['name'],
                    "home_score": e['homeScore'].get('display', 0) if 'homeScore' in e else 0,
                    "away_score": e['awayScore'].get('display', 0) if 'awayScore' in e else 0,
                    "slug": e['slug']
                }
                all_events.append(row)
            
            # Reduced delay
            time.sleep(random.uniform(0.1, 0.3))
            
        df = pd.DataFrame(all_events)
        if df.empty:
            print(f"Warning: No events found for Tournament {self.tournament_id} Season {self.season_id}. possibly blocked (403).")
            # Do not save empty file
            return df
            
        df.to_csv(schedule_path, index=False)
        print(f"Saved schedule ({len(df)} matches) to {schedule_path}")
        return df

    def get_match_data(self, event_id: int, force_update: bool = False) -> Dict:
        """
        Loads detailed match data (Lineups, Stats, Shotmap, Positions, Momentum, Odds).
        Checks local JSON/CSVs first, else fetches and saves.
        Returns a dict of DataFrames.
        """
        match_dir = self.events_dir / str(event_id)
        match_dir.mkdir(parents=True, exist_ok=True)
        
        # Files
        files = {
            'lineups': match_dir / "lineups.csv",
            'stats': match_dir / "stats.csv",
            'shots': match_dir / "shots.csv",
            'positions': match_dir / "positions.csv",
            'momentum': match_dir / "momentum.csv",
            'odds': match_dir / "odds.csv"
        }
        
        data = {}
        
        # Check if ALL primary files exist to skip fetching
        # If any major one is missing, we might want to refetch? 
        # For efficiency, check lineups (most critical).
        if files['lineups'].exists() and not force_update:
            for key, path in files.items():
                if path.exists():
                    data[key] = pd.read_csv(path)
            return data
            
        # --- Fetching ---
        
        # 1. Lineups (Includes Player Stats)
        df_lineups = create_lineups_df(event_id)
        if not df_lineups.empty:
            df_lineups.to_csv(files['lineups'], index=False)
        data['lineups'] = df_lineups
        
        # 2. Statistics (Team)
        df_stats = create_statistics_df(event_id)
        if not df_stats.empty:
            df_stats.to_csv(files['stats'], index=False)
        data['stats'] = df_stats
        
        # 3. Shotmap
        df_shots = create_shotmap_df(event_id)
        if not df_shots.empty:
            df_shots.to_csv(files['shots'], index=False)
        data['shots'] = df_shots
        
        # 4. Average Positions
        df_pos = create_average_positions_df(event_id)
        if not df_pos.empty:
            df_pos.to_csv(files['positions'], index=False)
        data['positions'] = df_pos
        
        # 5. Momentum (Graph)
        df_mom = create_graph_df(event_id)
        if not df_mom.empty:
            df_mom.to_csv(files['momentum'], index=False)
        data['momentum'] = df_mom
        
        # 6. Odds
        df_odds = create_odds_df(event_id)
        if not df_odds.empty:
            df_odds.to_csv(files['odds'], index=False)
        data['odds'] = df_odds

        # Minimal sleep between fetches in same match (practically atomic)
        # But between matches we might need a tiny breath
        return data

    def backfill_season(self, df_schedule: pd.DataFrame):
        """
        Iterates through the schedule and fetches missing match data.
        """
        print("Starting backfill for full season...")
        # Sort by date
        df_schedule = df_schedule.sort_values('startTimestamp')
        
        # Only finished matches need data usually? 
        # Or we want lineups for upcoming? (Lineups avail 1hr before)
        # For backtesting/training, we focus on 'finished'.
        
        finished = df_schedule[df_schedule['status'] == 'finished']
        
        for idx, row in tqdm(finished.iterrows(), total=len(finished), desc="Backfilling Matches"):
            self.get_match_data(row['id'])
            # Small delay to verify politeness, though create_* fns might sleep too?
            # inside create_* I removed sleeps in the module to control here?
            # Checked module: scrape_sofascore has no sleep.
            time.sleep(random.uniform(0.3, 1.0))
            
    def get_player_stats_from_matches(self, player_id: int, match_ids_before: List[int]) -> pd.DataFrame:
        """
        Reconstructs player rolling history by reading processed match files.
        This avoids 'downloading player log' by aggregating 'match lineups/stats'.
        Crucial for preventing data leakage: we only read matches appearing in 'match_ids_before'.
        """
        player_history = []
        
        for match_id in match_ids_before:
            lineups_path = self.events_dir / str(match_id) / "lineups.csv"
            if not lineups_path.exists():
                continue
            
            df_lineups = pd.read_csv(lineups_path)
            if 'player_id' in df_lineups.columns:
                player_row = df_lineups[df_lineups['player_id'] == player_id].copy()
                if not player_row.empty:
                    player_row['match_id'] = match_id
                    player_history.append(player_row)
        
        if not player_history:
            return pd.DataFrame()
            
        return pd.concat(player_history, ignore_index=True)
