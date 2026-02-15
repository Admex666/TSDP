import sys
import os
import json
import pandas as pd
from typing import List, Dict

# Setup path to import from modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.SofaScore_module import scrape_sofascore, create_odds_df

def fetch_nb1_season_data(tournament_id=187, season_id=61714, output_file="nb1_2024_25_data.json"):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                all_games = json.load(f)
            except json.JSONDecodeError:
                all_games = []
    else:
        all_games = []
    
    # Filter out games with 0 probabilities to re-fetch them
    all_games = [g for g in all_games if g.get("prob_home", 0) > 0]
    existing_ids = {game["id"] for game in all_games}
    
    # NB1 has 33 rounds
    for round_nr in range(1, 34):
        print(f"Checking round {round_nr}...")
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_nr}"
        data = scrape_sofascore(url)
        
        if not data or "events" not in data:
            print(f"No events found for round {round_nr}")
            continue
            
        for event in data["events"]:
            # Only process finished games
            if event["status"]["type"] != "finished":
                continue
                
            game_id = event["id"]
            if game_id in existing_ids:
                continue
                
            home_team = event["homeTeam"]["name"]
            away_team = event["awayTeam"]["name"]
            
            # Use display score or regular score
            home_score = event.get("homeScore", {}).get("display", 0)
            away_score = event.get("awayScore", {}).get("display", 0)
            
            # Outcome: 1 for Home Win (3 pts), X for Draw (1 pt), 2 for Away Win (0 pts for home)
            if home_score > away_score:
                winner = "Home"
                home_pts = 3
                away_pts = 0
            elif home_score < away_score:
                winner = "Away"
                home_pts = 0
                away_pts = 3
            else:
                winner = "Draw"
                home_pts = 1
                away_pts = 1
                
            # Get odds
            print(f"  Fetching odds for game {game_id}: {home_team} vs {away_team}")
            odds_df = create_odds_df(game_id)
            if odds_df.empty:
                print(f"  No odds for game {game_id}")
                continue
            
            # Robust mapping for choice names
            # Map "1" -> "Home", "X" -> "Draw", "2" -> "Away"
            # Or if they are already Home/Draw/Away
            probs = {}
            mapping = {"1": "Home", "X": "Draw", "2": "Away", "1": "Home", "X": "Draw", "2": "Away"}
            
            for _, o_row in odds_df.iterrows():
                name = str(o_row["name"])
                prob = o_row["prob_corr"]
                if name in mapping:
                    probs[mapping[name]] = prob
                elif "Home" in name:
                    probs["Home"] = prob
                elif "Draw" in name:
                    probs["Draw"] = prob
                elif "Away" in name:
                    probs["Away"] = prob
                else:
                    # If we can't map by name, use position if exactly 3 choices
                    pass
            
            # If still missing, try by position if 3 choices
            if len(probs) < 3 and len(odds_df) == 3:
                probs["Home"] = odds_df.iloc[0]["prob_corr"]
                probs["Draw"] = odds_df.iloc[1]["prob_corr"]
                probs["Away"] = odds_df.iloc[2]["prob_corr"]

            p_home = probs.get("Home", 0)
            p_draw = probs.get("Draw", 0)
            p_away = probs.get("Away", 0)

            if p_home == 0 and p_draw == 0 and p_away == 0:
                print(f"  Warning: Could not extract probabilities for {game_id}")
                continue

            # Expected points
            e_home_pts = p_home * 3 + p_draw * 1
            e_away_pts = p_away * 3 + p_draw * 1

            game_data = {
                "id": game_id,
                "round": round_nr,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "winner": winner,
                "home_pts": home_pts,
                "away_pts": away_pts,
                "e_home_pts": e_home_pts,
                "e_away_pts": e_away_pts,
                "prob_home": p_home,
                "prob_draw": p_draw,
                "prob_away": p_away
            }
            all_games.append(game_data)
            existing_ids.add(game_id)
            
            # Save after each game to be safe
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_games, f, ensure_ascii=False, indent=2)
            
    return all_games

if __name__ == "__main__":
    season_data = fetch_nb1_season_data()
    print(f"Total games collected: {len(season_data)}")
