"""
Data fetching module for SofaScore API
"""
import sys
import os
from typing import Dict, List, Optional, Any
import pandas as pd

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'modules'))
from SofaScore_module import scrape_sofascore, create_shotmap_df, fetch_passmap


class DataFetcher:
    """Handles all API data retrieval"""
    
    def __init__(self):
        self.cache = {}
    
    def get_round_matches(self, tournament_id: int, season_id: int, round_num: int) -> List[Dict]:
        """Get all matches for a specific round"""
        cache_key = f"matches_{tournament_id}_{season_id}_{round_num}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
        data = scrape_sofascore(url)
        
        matches = data.get('events', [])
        self.cache[cache_key] = matches
        return matches
    
    def get_team_statistics(self, team_id: int, tournament_id: int, season_id: int) -> Dict:
        """Get comprehensive team statistics for the season"""
        cache_key = f"stats_{team_id}_{tournament_id}_{season_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall"
        data = scrape_sofascore(url)
        
        stats = data.get('statistics', {})
        self.cache[cache_key] = stats
        return stats
    
    def get_team_form(self, team_id: int, limit: int = 10) -> List[Dict]:
        """Get team's recent matches"""
        cache_key = f"form_{team_id}_{limit}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/0"
        data = scrape_sofascore(url)
        
        events = data.get('events', [])[:limit]
        self.cache[cache_key] = events
        return events
    
    def get_top_players(self, team_id: int, tournament_id: int, season_id: int) -> Dict:
        """Get top players by various categories"""
        cache_key = f"players_{team_id}_{tournament_id}_{season_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/team/{team_id}/unique-tournament/{tournament_id}/season/{season_id}/top-players/overall"
        data = scrape_sofascore(url)
        
        top_players = data.get('topPlayers', {})
        self.cache[cache_key] = top_players
        return top_players
    
    def get_match_statistics(self, event_id: int) -> Dict:
        """Get statistics for a specific match"""
        cache_key = f"match_stats_{event_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
        data = scrape_sofascore(url)
        
        self.cache[cache_key] = data
        return data
    
    def get_shotmap(self, event_id: int) -> pd.DataFrame:
        """Get shot map data for a match"""
        cache_key = f"shotmap_{event_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        df = create_shotmap_df(event_id)
        self.cache[cache_key] = df
        return df
    
    def get_team_shotmaps(self, team_id: int, tournament_id: int, season_id: int, 
                          max_matches: int = 10) -> pd.DataFrame:
        """
        Aggregate shot maps from recent matches for a team
        """
        # Get recent matches
        form = self.get_team_form(team_id, limit=max_matches)
        
        all_shots = []
        
        for match in form:
            # Only get shotmaps for finished matches in the same tournament
            if match.get('status', {}).get('type') != 'finished':
                continue
            
            # Check if it's from the same tournament
            match_tournament_id = match.get('tournament', {}).get('uniqueTournament', {}).get('id')
            if match_tournament_id != tournament_id:
                continue
            
            event_id = match['id']
            
            try:
                shotmap_df = self.get_shotmap(event_id)
                
                if not shotmap_df.empty:
                    # Filter for this team
                    is_home = match['homeTeam']['id'] == team_id
                    team_shots = shotmap_df[shotmap_df['isHome'] == is_home].copy()
                    
                    if not team_shots.empty:
                        team_shots['event_id'] = event_id
                        all_shots.append(team_shots)
            except Exception as e:
                print(f"Error fetching shotmap for event {event_id}: {e}")
                continue
        
        if all_shots:
            return pd.concat(all_shots, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_player_passmap(self, event_id: int, player_id: int) -> pd.DataFrame:
        """Get pass map data for a specific player in a match"""
        cache_key = f"passmap_{event_id}_{player_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        df = fetch_passmap(event_id, player_id)
        self.cache[cache_key] = df
        return df
    
    def get_lineups(self, event_id: int) -> Dict:
        """Get lineups for a match"""
        cache_key = f"lineups_{event_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://www.sofascore.com/api/v1/event/{event_id}/lineups"
        data = scrape_sofascore(url)
        
        self.cache[cache_key] = data
        return data
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
