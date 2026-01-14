import time
import logging
import json
import os
import sys
from datetime import datetime
import pandas as pd
import joblib

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from riot_api import RiotEsportsAPI
from value_betting import ValueBettingEngine
from scrapers import OddsScraper
from discovery import TippmixDiscovery
import difflib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("live_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveScanner:
    def __init__(self):
        self.api = RiotEsportsAPI()
        self.odds_scraper = OddsScraper(headless=True)
        self.discovery = TippmixDiscovery(headless=True)
        
        # Discovery state
        self.odds_urls = {} # Mapping of riot_key -> tippmix_url
        self.last_discovery = 0
        self.discovery_interval = 600 # 10 minutes
        
        # Load models
        try:
            # Use absolute paths relative to root
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(root_dir, "models")
            
            self.gb_model = joblib.load(os.path.join(models_dir, "live_gb_model_20251031.joblib"))
            self.rf_model = joblib.load(os.path.join(models_dir, "live_rf_model_20251031.joblib"))
            self.scaler = joblib.load(os.path.join(models_dir, "live_scaler_20251031.joblib"))
            
            self.engine = ValueBettingEngine(
                self.gb_model, self.rf_model, self.scaler,
                min_edge=0.03, # 3% min edge
                min_confidence=0.4 # 40% min confidence
            )
            logger.info("✅ Models and ValueBettingEngine initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            sys.exit(1)

        self.active_games = {} # game_id -> {match_info, last_update}

    def scan_for_matches(self):
        """Fetch live matches and update active games list"""
        try:
            # 1. Periodic Discovery update
            now = time.time()
            if now - self.last_discovery > self.discovery_interval:
                logger.info("🔄 Running Tippmix discovery pass...")
                discovered_urls = self.discovery.discover_lol_matches()
                if discovered_urls:
                    self.odds_urls.update(discovered_urls)
                self.last_discovery = now

            live_events = self.api.get_live()
            if not live_events:
                # Fallback to schedule check
                schedule = self.api.get_schedule()
                live_events = [e for e in schedule if e.get('state') == 'inProgress']
            
            found_ids = set()
            for event in live_events:
                event_id = event.get('id')
                league_name = event.get('league', {}).get('name', 'Unknown')
                
                # Get details to find Game IDs
                details = self.api.get_event_details(event_id)
                if not details: continue
                
                match_data = details.get('match', {})
                teams = match_data.get('teams', [])
                if len(teams) < 2: continue
                
                riot_blue = teams[0].get('name', teams[0].get('code', 'Unknown'))
                riot_red = teams[1].get('name', teams[1].get('code', 'Unknown'))
                
                games = match_data.get('games', [])
                for g in games:
                    if g.get('state') == 'inProgress':
                        gid = g['id']
                        found_ids.add(gid)
                        
                        if gid not in self.active_games:
                            logger.info(f"✨ New Live Game Detected: {league_name} | {riot_blue} vs {riot_red} (ID: {gid})")
                            self.active_games[gid] = {
                                'league': league_name,
                                'blue': riot_blue,
                                'red': riot_red,
                                'event_id': event_id,
                                'start_time': datetime.now().isoformat()
                            }
            
            # Remove finished games
            finished = [gid for gid in self.active_games if gid not in found_ids]
            for gid in finished:
                logger.info(f"🏁 Game Finished: {gid}")
                del self.active_games[gid]
                
        except Exception as e:
            logger.error(f"Error during scan: {e}")

    def run(self):
        logger.info("🚀 Live Scanner started. Press Ctrl+C to stop.")
        
        # In a real scenario, we'd need a mapping for Tippmix URLs
        # For now, let's assume we can find the odds if we have a base URL or search
        # Since the user didn't specify a way to find URLs, we will log probabilities
        
        while True:
            self.scan_for_matches()
            
            if not self.active_games:
                logger.info("No active games found. Sleeping 60s...")
                time.sleep(60)
                continue
                
            for gid, info in self.active_games.items():
                try:
                    match_state = self.api.get_latest_match_state(gid)
                    if not match_state: continue
                    
                    # Calculate win probability
                    features = self.engine.calculate_features(match_state)
                    prob_blue, prob_red = self.engine.predict_win_probability(features)
                    
                    logger.info(f"📊 [{info['league']}] {info['blue']} {match_state['blue_team']['kills']} - {match_state['red_team']['kills']} {info['red']} | "
                                f"Time: {match_state['game_time']} | "
                                f"BLUE Win%: {prob_blue:.1%}, RED Win%: {prob_red:.1%}")
                    
                    # Scrape Odds if URL is known
                    # Note: We need a way to link GID to Tippmix URL. 
                    # For this demo, we can check if there's a file called 'odds_mapping.json'
                    odds_url = self.get_odds_url(info)
                    value_bets = []
                    if odds_url:
                        odds_data = self.odds_scraper.scrape(odds_url)
                        if odds_data:
                            # Auto-map teams
                            tippmix_teams = []
                            if odds_data['markets']:
                                tippmix_teams = [opt['name'] for opt in odds_data['markets'][0]['options']]
                            
                            home_is_blue = self.engine.auto_map_teams(info['blue'], info['red'], tippmix_teams)
                            
                            value_bets = self.engine.find_value_bets(
                                match_state, odds_data, 
                                home_is_blue=home_is_blue
                            )
                            
                            for vb in value_bets:
                                logger.info(f"🎯 VALUE FOUND: {vb['team_name']} @ {vb['odds']} (Edge: {vb['edge']:.1f}%)")

                    # Log to CSV
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'game_id': gid,
                        'league': info['league'],
                        'blue_team': info['blue'],
                        'red_team': info['red'],
                        'game_time': match_state['game_time'],
                        'blue_kills': match_state['blue_team']['kills'],
                        'red_kills': match_state['red_team']['kills'],
                        'gold_diff': match_state['blue_team']['gold'] - match_state['red_team']['gold'],
                        'prob_blue': prob_blue,
                        'prob_red': prob_red,
                        'value_bets': json.dumps(value_bets)
                    }
                    self.save_record(record)
                    
                except Exception as e:
                    logger.error(f"Error processing game {gid}: {e}")
            
            time.sleep(30) # Wait 30s between updates

    def get_odds_url(self, info):
        """Try to map match to a Tippmix URL automatically"""
        riot_t1 = info['blue'].lower()
        riot_t2 = info['red'].lower()
        
        # Exact match check
        key = f"{info['blue']}_vs_{info['red']}".replace(" ", "_")
        if key in self.odds_urls:
            return self.odds_urls[key]
            
        # Fuzzy match check
        best_ratio = 0
        best_url = None
        
        for tippmix_key, url in self.odds_urls.items():
            # tippmix_key is usually t1_vs_t2
            if "_vs_" not in tippmix_key: continue
            try:
                tp1, tp2 = tippmix_key.lower().split("_vs_")
                
                # Simple average ratio of t1 matching t1 and t2 matching t2
                r1 = difflib.SequenceMatcher(None, riot_t1, tp1).ratio()
                r2 = difflib.SequenceMatcher(None, riot_t2, tp2).ratio()
                
                # Also check inverted (t1 matching t2)
                r1_inv = difflib.SequenceMatcher(None, riot_t1, tp2).ratio()
                r2_inv = difflib.SequenceMatcher(None, riot_t2, tp1).ratio()
                
                ratio = max((r1 + r2) / 2, (r1_inv + r2_inv) / 2)
                
                if ratio > best_ratio and ratio > 0.6: # 60% threshold
                    best_ratio = ratio
                    best_url = url
            except:
                continue
                
        if best_url:
            logger.info(f"🔗 Auto-linked {info['blue']} vs {info['red']} to Tippmix (Match confidence: {best_ratio:.1%})")
            return best_url
            
        # Fallback to manual mapping if file exists
        mapping_file = "odds_mapping.json"
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    return mapping.get(key)
            except: pass
        return None

    def save_record(self, record):
        file_path = "live_scanner_history.csv"
        df = pd.DataFrame([record])
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    scanner = LiveScanner()
    scanner.run()
