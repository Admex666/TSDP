import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))

from SofaScore_module import scrape_sofascore
import json

def explore_premier_league_data():
    """Explore available data for Premier League matches"""
    
    # Get upcoming matches for round 23
    print("=" * 80)
    print("EXPLORING PREMIER LEAGUE ROUND 23 DATA")
    print("=" * 80)
    
    url = "https://www.sofascore.com/api/v1/unique-tournament/17/season/76986/events/round/23"
    data = scrape_sofascore(url)
    
    if not data or 'events' not in data:
        print("Failed to fetch data")
        return
    
    events = data['events']
    print(f"\nFound {len(events)} matches in Round 23\n")
    
    # Pick first match for detailed exploration
    if events:
        match = events[0]
        event_id = match['id']
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        
        print(f"Exploring: {home_team} vs {away_team} (ID: {event_id})")
        print("=" * 80)
        
        # Get team IDs
        home_team_id = match['homeTeam']['id']
        away_team_id = match['awayTeam']['id']
        season_id = match['season']['id']
        tournament_id = match['tournament']['uniqueTournament']['id']
        
        print(f"\nHome Team ID: {home_team_id}")
        print(f"Away Team ID: {away_team_id}")
        print(f"Season ID: {season_id}")
        print(f"Tournament ID: {tournament_id}")
        
        # Explore available endpoints
        endpoints = {
            "Team Statistics (Home)": f"https://www.sofascore.com/api/v1/team/{home_team_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall",
            "Team Form (Home)": f"https://www.sofascore.com/api/v1/team/{home_team_id}/events/last/0",
            "Team Players (Home)": f"https://www.sofascore.com/api/v1/team/{home_team_id}/unique-tournament/{tournament_id}/season/{season_id}/top-players/overall",
            "H2H": f"https://www.sofascore.com/api/v1/event/{event_id}/h2h/events",
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            print(f"\n{'=' * 80}")
            print(f"Testing: {name}")
            print(f"URL: {endpoint}")
            print(f"{'=' * 80}")
            
            result = scrape_sofascore(endpoint)
            
            if result:
                print(f"✓ Success!")
                
                # Show structure
                if isinstance(result, dict):
                    print(f"Keys: {list(result.keys())[:10]}")
                    
                    # Show sample data for statistics
                    if 'statistics' in result:
                        stats = result['statistics']
                        if isinstance(stats, dict):
                            print(f"\nStatistics groups: {list(stats.keys())[:10]}")
                            # Show first group details
                            first_key = list(stats.keys())[0] if stats else None
                            if first_key:
                                print(f"\nSample from '{first_key}':")
                                print(json.dumps(stats[first_key], indent=2)[:500])
                    
                    # Show sample for top players
                    if 'topPlayers' in result:
                        print(f"\nTop players categories: {list(result['topPlayers'].keys())[:10]}")
                    
                    # Show sample for events
                    if 'events' in result:
                        print(f"\nNumber of events: {len(result['events'])}")
                        if result['events']:
                            print(f"Sample event keys: {list(result['events'][0].keys())[:15]}")
                
                results[name] = result
            else:
                print(f"✗ Failed")
                results[name] = None
        
        # Save results
        output_file = os.path.join(os.path.dirname(__file__), 'api_exploration.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")
        print(f"{'=' * 80}")

if __name__ == "__main__":
    explore_premier_league_data()
