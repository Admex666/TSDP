import os
import sys
import time
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

# Add modules directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../modules')))

from SofaScore_module import scrape_sofascore, create_lineups_df, create_player_stats_df
from db_schema import SessionLocal, init_db, Tournament, Team, Player, Match, PlayerMatchStats
from fantasy_calculator import calculate_fantasy_points, get_position_category

# Configuration
TOURNAMENT_ID = 17    # Premier League
SEASON_ID = 61627     # 24/25 Season (or previous as requested)
UNIQUE_TOURNAMENT_ID = 17

def get_matches_for_round(unique_tournament_id, season_id, round_num):
    """Fetch basic match info for a specific round"""
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{unique_tournament_id}/season/{season_id}/events/round/{round_num}"
    data = scrape_sofascore(url)
    return data.get('events', [])

def process_match(session: Session, match_data: dict, round_num: int):
    match_id = match_data['id']
    print(f"Processing Match: {match_data['homeTeam']['name']} vs {match_data['awayTeam']['name']} (ID: {match_id})")
    
    # Check if match exists
    existing_match = session.query(Match).filter_by(id=match_id).first()
    if existing_match:
        print(f"  Match {match_id} exists. Updating stats...")
        # Optional: Delete existing stats for this match to perform clean re-insert
        session.query(PlayerMatchStats).filter_by(match_id=match_id).delete()
        session.commit()
        # We don't return here, we proceed to re-process
        
    # Ensure Teams Exist (same as before)
    for team_key in ['homeTeam', 'awayTeam']:
        t_data = match_data[team_key]
        team = session.query(Team).filter_by(id=t_data['id']).first()
        if not team:
            team = Team(id=t_data['id'], name=t_data['name'], short_name=t_data.get('shortName'))
            session.add(team)
            session.commit()

    # Ensure Tournament Exists (same as before)
    tourn_data = match_data['tournament']
    tournament = session.query(Tournament).filter_by(id=tourn_data['id']).first()
    if not tournament:
        tournament = Tournament(id=tourn_data['id'], name=tourn_data['name'], slug=tourn_data['slug'])
        session.add(tournament)
        session.commit()

    # Create/Update Match
    if not existing_match:
        match = Match(
            id=match_id,
            tournament_id=tourn_data['id'],
            season_id=SEASON_ID,
            round=round_num,
            date=datetime.fromtimestamp(match_data['startTimestamp']),
            home_team_id=match_data['homeTeam']['id'],
            away_team_id=match_data['awayTeam']['id'],
            home_score=match_data['homeScore']['current'],
            away_score=match_data['awayScore']['current']
        )
        session.add(match)
    
    match_data_home_id = match_data['homeTeam']['id']
    match_data_away_id = match_data['awayTeam']['id']
    
    # --- Process Players & Stats ---
    # We use scraping functions from SofaScore_module
    
    # ... (rest of logic same) ...
    
    url_lineups = f"https://www.sofascore.com/api/v1/event/{match_id}/lineups"
    lineups_data = scrape_sofascore(url_lineups)
    
    players_to_add = []
    
    if 'home' in lineups_data and 'players' in lineups_data['home']:
        process_team_players(session, lineups_data['home']['players'], match_id, match_data_home_id, players_to_add)
        
    if 'away' in lineups_data and 'players' in lineups_data['away']:
        process_team_players(session, lineups_data['away']['players'], match_id, match_data_away_id, players_to_add)
        
    session.add_all(players_to_add)
    session.commit()
    print(f"  Saved/Updated {len(players_to_add)} player stats records.")

def process_team_players(session, players_list, match_id, team_id, players_to_add_list):
    for p_entry in players_list:
        p_data = p_entry['player']
        stats = p_entry.get('statistics', {})
        
        # Ensure Player Exists
        player = session.query(Player).filter_by(id=p_data['id']).first()
        if not player:
            # Try to get position from entry
            pos = p_entry.get('position', 'F') # default
            player = Player(id=p_data['id'], name=p_data['name'], slug=p_data['slug'], position=pos)
            session.add(player)
            session.commit()
        
        # Only process if they played (have stats or minutes)
        minutes = stats.get('minutesPlayed', 0)
        if minutes > 0 or p_entry.get('substitute') is False: # Start or played sub
            
            # Calculate Fantasy Points
            fp = calculate_fantasy_points(stats, player.position)
            
            p_stats = PlayerMatchStats(
                player_id=player.id,
                match_id=match_id,
                team_id=team_id,
                minutes=minutes,
                rating=stats.get('rating'),
                goals=stats.get('goals', 0),
                assists=stats.get('assists', 0),
                total_points=fp,
                detailed_stats=stats
            )
            players_to_add_list.append(p_stats)

def run_pipeline():
    print("Initializing Database...")
    init_db()
    session = SessionLocal()
    
    # Process Rounds (1-38)
    for round_num in range(1, 39): 
        print(f"\n--- Processing Round {round_num} ---")
        matches = get_matches_for_round(UNIQUE_TOURNAMENT_ID, SEASON_ID, round_num)
        
        for match in matches:
            if match['status']['type'] == 'finished':
                try:
                    process_match(session, match, round_num)
                except Exception as e:
                    print(f"Error processing match {match.get('id')}: {e}")
                    session.rollback()
        
    session.close()
    print("\nPipeline Finished.")

if __name__ == "__main__":
    run_pipeline()
