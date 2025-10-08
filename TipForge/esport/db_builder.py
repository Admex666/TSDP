"""
Esports Devs API - Hierarchikus adatgyűjtő szkript
Gyűjti: Leagues → Tournaments/Seasons → Matches → Games
"""

import requests
import pandas as pd
import time
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# API konfiguráció
BASE_URL = "https://esports-devs.p.rapidapi.com"
API_KEY = os.getenv('API_KEY')  # .env-ből vagy beégetett kulcs

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "esports-devs.p.rapidapi.com"
}

# Rate limiting
RATE_LIMIT_DELAY = 0.1  # másodperc API-hívások között
MAX_RETRIES = 3


class EsportsDataCollector:
    """Esports adatok gyűjtője és mentője"""
    
    def __init__(self, output_dir: str = "esport/output", cs_only: bool = True):
        self.output_dir = output_dir
        self.cs_class_id = "112"  # Counter-Strike 2
        self.cs_only = cs_only
        
        # CSV fájlnevek
        self.csv_files = {
            'leagues': os.path.join(output_dir, 'leagues.csv'),
            'tournaments': os.path.join(output_dir, 'tournaments.csv'),
            'seasons': os.path.join(output_dir, 'seasons.csv'),
            'matches': os.path.join(output_dir, 'matches.csv'),
            'games': os.path.join(output_dir, 'games.csv')
        }
        
        # Számlálók
        self.counters = {
            'leagues': 0,
            'tournaments': 0,
            'seasons': 0,
            'matches': 0,
            'games': 0
        }
        
        # Létrehozzuk az output mappát
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializáljuk a CSV fájlokat üres fejlécekkel
        self._init_csv_files()
        
    def _init_csv_files(self):
        """CSV fájlok inicializálása fejlécekkel"""
        
        # Leagues
        pd.DataFrame(columns=['id', 'name', 'class_id']).to_csv(
            self.csv_files['leagues'], index=False
        )
        
        # Tournaments
        pd.DataFrame(columns=['id', 'league_id', 'name']).to_csv(
            self.csv_files['tournaments'], index=False
        )
        
        # Seasons
        pd.DataFrame(columns=['id', 'league_id', 'name']).to_csv(
            self.csv_files['seasons'], index=False
        )
        
        # Matches
        pd.DataFrame(columns=[
            'id', 'tournament_id', 'season_id', 
            'home_team_id', 'home_team_name',
            'away_team_id', 'away_team_name',
            'home_team_score', 'away_team_score',
            'start_time'
        ]).to_csv(self.csv_files['matches'], index=False)
        
        # Games
        pd.DataFrame(columns=[
            'id', 'match_id', 'map',
            'home_team_score', 'away_team_score',
            'has_statistics', 'has_rounds', 'has_lineups'
        ]).to_csv(self.csv_files['games'], index=False)
    
    def _append_to_csv(self, filename: str, data: List[Dict]):
        """Adatok hozzáfűzése CSV fájlhoz"""
        if not data:
            return
        
        df = pd.DataFrame(data)
        df.to_csv(filename, mode='a', header=False, index=False)
    
    def _api_call(self, endpoint: str, params: Dict) -> Optional[List[Dict]]:
        """Általános API hívás retry logikával"""
        url = f"{BASE_URL}/{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                response = requests.get(url, headers=HEADERS, params=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Rate limit, várakozás {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API hiba: {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"❌ Hiba történt: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    return None
        
        return None
    
    def get_leagues(self, limit: int = 100) -> List[Dict]:
        """Ligák lekérése (CS2-re szűrve)"""
        print("📋 Ligák lekérése...")
        
        params = {"limit": limit, "offset": 0}
        if self.cs_only:
            params["class_id"] = f"eq.{self.cs_class_id}"
        
        data = self._api_call("leagues", params)
        
        if data:
            print(f"✅ {len(data)} liga lekérve")
            return data
        return []
    
    def get_tournaments(self, league_id: int, limit: int = 100) -> List[Dict]:
        """Tornák lekérése egy ligához"""
        params = {
            "league_id": f"eq.{league_id}",
            "limit": limit,
            "offset": 0
        }
        
        data = self._api_call("tournaments", params)
        return data if data else []
    
    def get_seasons(self, league_id: int, limit: int = 100) -> List[Dict]:
        """Szezonok lekérése egy ligához"""
        params = {
            "league_id": f"eq.{league_id}",
            "limit": limit,
            "offset": 0
        }
        
        data = self._api_call("seasons", params)
        return data if data else []
    
    def get_matches(self, tournament_id: Optional[int] = None, 
                   season_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Meccsek lekérése tournament vagy season alapján"""
        params = {"limit": limit, "offset": 0}
        
        if tournament_id:
            params["tournament_id"] = f"eq.{tournament_id}"
        elif season_id:
            params["season_id"] = f"eq.{season_id}"
        else:
            return []
        
        data = self._api_call("matches", params)
        return data if data else []
    
    def get_games(self, match_id: int, limit: int = 50) -> List[Dict]:
        """Játékok (mapok) lekérése egy meccshez"""
        params = {
            "match_id": f"eq.{match_id}",
            "limit": limit,
            "offset": 0
        }
        
        data = self._api_call("matches-games", params)
        return data if data else []
    
    def flatten_league(self, league: Dict) -> Dict:
        """Liga adatok egyszerűsítése"""
        return {
            'id': league.get('id'),
            'name': league.get('name'),
            'class_id': league.get('class_id')
        }
    
    def flatten_tournament(self, tournament: Dict, league_id: int) -> Dict:
        """Tournament adatok egyszerűsítése"""
        return {
            'id': tournament.get('id'),
            'league_id': league_id,
            'name': tournament.get('name')
        }
    
    def flatten_season(self, season: Dict, league_id: int) -> Dict:
        """Szezon adatok egyszerűsítése"""
        return {
            'id': season.get('id'),
            'league_id': league_id,
            'name': season.get('name')
        }
    
    def flatten_match(self, match: Dict, tournament_id: Optional[int] = None,
                     season_id: Optional[int] = None) -> Dict:
        """Meccs adatok egyszerűsítése"""
        return {
            'id': match.get('id'),
            'tournament_id': tournament_id,
            'season_id': season_id,
            'home_team_id': match.get('home_team_id'),
            'home_team_name': match.get('home_team_name'),
            'away_team_id': match.get('away_team_id'),
            'away_team_name': match.get('away_team_name'),
            'home_team_score': match.get('home_team_score', {}).get('current'),
            'away_team_score': match.get('away_team_score', {}).get('current'),
            'start_time': match.get('start_time')
        }
    
    def flatten_game(self, game: Dict, match_id: int) -> Dict:
        """Game (map) adatok egyszerűsítése"""
        return {
            'id': game.get('id'),
            'match_id': match_id,
            'map': game.get('map'),
            'home_team_score': game.get('home_team_score', {}).get('display'),
            'away_team_score': game.get('away_team_score', {}).get('display'),
            'has_statistics': game.get('has_statistics'),
            'has_rounds': game.get('has_rounds'),
            'has_lineups': game.get('has_lineups')
        }
    
    def save_leagues(self, leagues: List[Dict]):
        """Ligák mentése azonnal"""
        data = [self.flatten_league(l) for l in leagues]
        self._append_to_csv(self.csv_files['leagues'], data)
        self.counters['leagues'] += len(data)
    
    def save_tournaments(self, tournaments: List[Dict], league_id: int):
        """Tornák mentése azonnal"""
        data = [self.flatten_tournament(t, league_id) for t in tournaments]
        self._append_to_csv(self.csv_files['tournaments'], data)
        self.counters['tournaments'] += len(data)
    
    def save_seasons(self, seasons: List[Dict], league_id: int):
        """Szezonok mentése azonnal"""
        data = [self.flatten_season(s, league_id) for s in seasons]
        self._append_to_csv(self.csv_files['seasons'], data)
        self.counters['seasons'] += len(data)
    
    def save_matches(self, matches: List[Dict], tournament_id: Optional[int] = None,
                    season_id: Optional[int] = None):
        """Meccsek mentése azonnal"""
        data = [self.flatten_match(m, tournament_id, season_id) for m in matches]
        self._append_to_csv(self.csv_files['matches'], data)
        self.counters['matches'] += len(data)
    
    def save_games(self, games: List[Dict], match_id: int):
        """Játékok mentése azonnal"""
        data = [self.flatten_game(g, match_id) for g in games]
        self._append_to_csv(self.csv_files['games'], data)
        self.counters['games'] += len(data)
    
    def collect_all_data(self, max_leagues: Optional[int] = None):
        """Teljes hierarchikus adatgyűjtés folyamatos mentéssel"""
        print("\n🚀 Esports adatgyűjtés indítása...\n")
        start_time = datetime.now()
        
        # 1. Ligák
        leagues = self.get_leagues(limit=max_leagues or 100)
        if max_leagues:
            leagues = leagues[:max_leagues]
        
        self.save_leagues(leagues)
        print(f"💾 {len(leagues)} liga mentve\n")
        
        for league in tqdm(leagues, desc="Ligák feldolgozása"):
            league_id = league['id']
            
            # 2. Tornák
            tournaments = self.get_tournaments(league_id)
            if tournaments:
                self.save_tournaments(tournaments, league_id)
                
                for tournament in tournaments:
                    tournament_id = tournament['id']
                    
                    # 3. Meccsek (tournament alapján)
                    matches = self.get_matches(tournament_id=tournament_id)
                    if matches:
                        self.save_matches(matches, tournament_id=tournament_id)
                        
                        for match in matches:
                            match_id = match['id']
                            
                            # 4. Játékok (mapok)
                            games = self.get_games(match_id)
                            if games:
                                self.save_games(games, match_id)
            
            # 5. Szezonok
            seasons = self.get_seasons(league_id)
            if seasons:
                self.save_seasons(seasons, league_id)
                
                for season in seasons:
                    season_id = season['id']
                    
                    # 6. Meccsek (season alapján)
                    matches = self.get_matches(season_id=season_id)
                    if matches:
                        self.save_matches(matches, season_id=season_id)
                        
                        for match in matches:
                            match_id = match['id']
                            
                            games = self.get_games(match_id)
                            if games:
                                self.save_games(games, match_id)
        
        elapsed = datetime.now() - start_time
        self._print_summary(elapsed)
    
    def _print_summary(self, elapsed):
        """Összefoglaló kiírása"""
        print("\n" + "="*60)
        print("📊 ADATGYŰJTÉS BEFEJEZVE")
        print("="*60)
        print(f"⏱️  Futási idő: {elapsed}")
        print(f"📋 Ligák: {self.counters['leagues']}")
        print(f"🏆 Tornák: {self.counters['tournaments']}")
        print(f"📅 Szezonok: {self.counters['seasons']}")
        print(f"⚔️  Meccsek: {self.counters['matches']}")
        print(f"🗺️  Játékok (mapok): {self.counters['games']}")
        print("="*60 + "\n")


def main():
    """Fő program"""
    parser = argparse.ArgumentParser(description='Esports Devs API adatgyűjtő')
    parser.add_argument('--limit', type=int, help='Ligák maximális száma', default=None)
    parser.add_argument('--output', type=str, help='Output mappa', default='output')
    parser.add_argument('--all-games', action='store_true', help='Minden játék, nem csak CS2')
    
    args = parser.parse_args()
    
    # Adatgyűjtés
    collector = EsportsDataCollector(
        output_dir=args.output,
        cs_only=not args.all_games
    )
    
    collector.collect_all_data(max_leagues=args.limit)
    
    print("✨ Adatgyűjtés sikeresen befejezve!\n")


if __name__ == "__main__":
    main()