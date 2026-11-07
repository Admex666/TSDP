"""
Europa League 2026/27 (679 #id:96522) Match & Odds Downloader
Fetches all matches (League Phase & Qualification) with final/current scores,
status, start times, and comprehensive betting odds (1X2, Over/Under 2.5, BTTS).
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add TSDP modules path
MODULES_PATH = r"E:\Data\TSDP\modules"
if MODULES_PATH not in sys.path:
    sys.path.append(MODULES_PATH)

from SofaScore_module import scrape_sofascore


def parse_fraction(val):
    """Convert fractional odds ('67/100') to decimal odds (1.67)."""
    if not val or "/" not in str(val):
        return None
    try:
        num, denom = str(val).split("/")
        return round(int(num) / int(denom) + 1.0, 3)
    except Exception:
        return None


def extract_match_odds(event_id: int):
    """
    Extract closing and opening odds (1X2, Over/Under 2.5, BTTS)
    and margin-corrected probabilities for a given event ID.
    """
    url = f"https://www.sofascore.com/api/v1/event/{event_id}/odds/1/all"
    data = scrape_sofascore(url)
    
    odds_res = {
        "odds_1": None,
        "odds_X": None,
        "odds_2": None,
        "init_odds_1": None,
        "init_odds_X": None,
        "init_odds_2": None,
        "prob_corr_1": None,
        "prob_corr_X": None,
        "prob_corr_2": None,
        "odds_over_2_5": None,
        "odds_under_2_5": None,
        "odds_btts_yes": None,
        "odds_btts_no": None,
    }
    
    if not data or not data.get("markets"):
        return odds_res
        
    for m in data["markets"]:
        group = m.get("marketGroup")
        name = m.get("marketName")
        cgroup = str(m.get("choiceGroup"))
        choices = m.get("choices", [])
        
        # 1X2 Full Time
        if group == "1X2" and name == "Full time":
            probs = {}
            for c in choices:
                cname = c.get("name")
                odd_val = parse_fraction(c.get("fractionalValue"))
                init_val = parse_fraction(c.get("initialFractionalValue"))
                if cname in ["1", "X", "2"]:
                    odds_res[f"odds_{cname}"] = odd_val
                    odds_res[f"init_odds_{cname}"] = init_val
                    if odd_val and odd_val > 0:
                        probs[cname] = 1.0 / odd_val
                        
            sum_prob = sum(probs.values())
            if sum_prob > 0:
                for k in ["1", "X", "2"]:
                    if k in probs:
                        odds_res[f"prob_corr_{k}"] = round(probs[k] / sum_prob, 4)
                        
        # Over / Under 2.5
        elif group == "Match goals" and cgroup == "2.5":
            for c in choices:
                if c.get("name") == "Over":
                    odds_res["odds_over_2_5"] = parse_fraction(c.get("fractionalValue"))
                elif c.get("name") == "Under":
                    odds_res["odds_under_2_5"] = parse_fraction(c.get("fractionalValue"))
                    
        # Both teams to score
        elif group == "Both teams to score":
            for c in choices:
                if c.get("name") == "Yes":
                    odds_res["odds_btts_yes"] = parse_fraction(c.get("fractionalValue"))
                elif c.get("name") == "No":
                    odds_res["odds_btts_no"] = parse_fraction(c.get("fractionalValue"))
                    
    return odds_res


def get_tournament_rounds(tournament_id: int = 679, season_id: int = 96522):
    """Fetch all round descriptors for tournament and season."""
    url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"
    rounds_data = scrape_sofascore(url)
    if not rounds_data or "rounds" not in rounds_data:
        return []
    return rounds_data["rounds"]


def build_round_url(tournament_id: int, season_id: int, round_info: dict) -> str:
    """Construct SofaScore API endpoint for a specific round."""
    rnd = round_info["round"]
    slug = round_info.get("slug")
    prefix = round_info.get("prefix")
    
    if prefix and slug:
        return f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{rnd}/slug/{slug}/prefix/{prefix}"
    elif slug:
        return f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{rnd}/slug/{slug}"
    else:
        return f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{rnd}"


def scrape_all_europa_league_matches(
    tournament_id: int = 679,
    season_id: int = 96522,
    output_csv: str = "europa_league_2026_matches.csv",
    include_qualifications: bool = True
):
    """
    Download all matches and odds for the Europa League season and export to CSV.
    """
    print(f"=== Starting Europa League Scraper (Tournament {tournament_id}, Season {season_id}) ===")
    
    # 1. Load existing CSV if available for incremental update/resume
    existing_matches = {}
    if os.path.exists(output_csv):
        try:
            df_old = pd.read_csv(output_csv)
            for _, row in df_old.iterrows():
                existing_matches[int(row["match_id"])] = row.to_dict()
            print(f"Found existing CSV with {len(existing_matches)} matches. Will update/resume.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting fresh.")

    # 2. Fetch rounds list
    rounds = get_tournament_rounds(tournament_id, season_id)
    if not rounds:
        print("Error: Could not fetch tournament rounds!")
        return pd.DataFrame()
        
    print(f"Discovered {len(rounds)} tournament rounds.")

    all_matches_data = dict(existing_matches)
    
    for r_entry in rounds:
        round_nr = r_entry["round"]
        round_name = r_entry.get("name", f"Round {round_nr}")
        is_qualification = "qualification" in round_name.lower() or "playoff" in round_name.lower() or "slug" in r_entry
        
        if is_qualification and not include_qualifications:
            continue
            
        stage_name = "Qualification" if is_qualification else "League Phase"
        round_url = build_round_url(tournament_id, season_id, r_entry)
        
        events_resp = scrape_sofascore(round_url)
        events = events_resp.get("events", []) if events_resp else []
        
        if not events:
            continue

        desc = f"{stage_name} - {round_name}"
        for ev in tqdm(events, desc=f"Scraping {desc}", leave=False):
            match_id = ev.get("id")
            if not match_id:
                continue
                
            # If match was already completed and has odds, we can reuse it
            if match_id in all_matches_data:
                cached = all_matches_data[match_id]
                # If finished and odds already populated, keep cached
                if cached.get("status_type") == "finished" and pd.notna(cached.get("odds_1")):
                    continue

            # Basic timestamps
            start_ts = ev.get("startTimestamp")
            start_dt_str = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S") if start_ts else None
            
            # Teams
            home_team = ev.get("homeTeam", {})
            away_team = ev.get("awayTeam", {})
            
            # Scores
            home_score_dict = ev.get("homeScore", {})
            away_score_dict = ev.get("awayScore", {})
            
            home_score = home_score_dict.get("display", home_score_dict.get("current"))
            away_score = away_score_dict.get("display", away_score_dict.get("current"))
            home_score_ht = home_score_dict.get("period1")
            away_score_ht = away_score_dict.get("period1")
            
            status_obj = ev.get("status", {})
            status_desc = status_obj.get("description")
            status_type = status_obj.get("type")
            
            # Odds
            odds_info = extract_match_odds(match_id)
            
            match_row = {
                "match_id": match_id,
                "tournament_id": tournament_id,
                "tournament_name": ev.get("tournament", {}).get("name", "UEFA Europa League"),
                "season_id": season_id,
                "season_name": ev.get("season", {}).get("name", "26/27"),
                "stage": stage_name,
                "round_name": round_name,
                "round_nr": round_nr,
                "start_timestamp": start_ts,
                "start_time_utc": start_dt_str,
                "status": status_desc,
                "status_type": status_type,
                "winner_code": ev.get("winnerCode"),
                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name"),
                "home_team_short": home_team.get("shortName"),
                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name"),
                "away_team_short": away_team.get("shortName"),
                "home_score": home_score,
                "away_score": away_score,
                "home_score_ht": home_score_ht,
                "away_score_ht": away_score_ht,
                **odds_info
            }
            
            all_matches_data[match_id] = match_row
            
        # Periodic intermediate save after each round
        df_inter = pd.DataFrame(list(all_matches_data.values()))
        if not df_inter.empty:
            df_inter.sort_values(by=["start_timestamp", "match_id"], inplace=True)
            df_inter.to_csv(output_csv, index=False, encoding="utf-8-sig")

    df_final = pd.DataFrame(list(all_matches_data.values()))
    if not df_final.empty:
        df_final.sort_values(by=["start_timestamp", "match_id"], inplace=True)
        df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n[Success] Scraped total {len(df_final)} matches. Saved to '{output_csv}'.")
    else:
        print("\n[Warning] No matches were scraped.")
        
    return df_final


if __name__ == "__main__":
    df = scrape_all_europa_league_matches()
    print("\nSummary of scraped matches:")
    if not df.empty:
        print(f"Total matches: {len(df)}")
        print(f"Status distribution:\n{df['status'].value_counts()}")
        print(f"Stages:\n{df['stage'].value_counts()}")
        print(f"Matches with 1X2 odds: {df['odds_1'].notna().sum()}/{len(df)}")
