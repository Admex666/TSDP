import json
import pandas as pd
import numpy as np
import os

def norm_team(name):
    if not name:
        return ""
    name = name.strip()
    # Align names between IMDb and SofaScore
    if name == "Korea Republic":
        return "South Korea"
    if name == "United States":
        return "USA"
    return name

def get_stat(stats_data, key_name):
    if not stats_data or "statistics" not in stats_data:
        return 0, 0
    for period in stats_data.get("statistics", []):
        if period.get("period") == "ALL":
            for group in period.get("groups", []):
                for item in group.get("statisticsItems", []):
                    if item.get("key") == key_name:
                        # Return float if expectedGoals, else try int, else default to 0
                        h_val = item.get("homeValue")
                        a_val = item.get("awayValue")
                        try:
                            return float(h_val) if h_val is not None else 0.0, float(a_val) if a_val is not None else 0.0
                        except ValueError:
                            return 0.0, 0.0
    return 0.0, 0.0

def main():
    imdb_path = r"C:\Users\Adam\.gemini\antigravity-ide\brain\5c011315-aa8e-4344-961d-f9230926d956\scratch\all_imdb_episodes_deduped.csv"
    sofascore_path = r"e:\Data\TSDP\scripts\WC_imdb\sofascore_data.json"
    output_path = r"e:\Data\TSDP\scripts\WC_imdb\WC_matches_final.csv"

    print("Loading data...")
    imdb_df = pd.read_csv(imdb_path)
    with open(sofascore_path, "r", encoding="utf-8") as f:
        ss_data = json.load(f)

    # Convert SofaScore matches dictionary for fast lookup
    ss_matches = []
    for m_id, m in ss_data["matches"].items():
        details = m.get("details", {}).get("event", {})
        home = norm_team(details.get("homeTeam", {}).get("name"))
        away = norm_team(details.get("awayTeam", {}).get("name"))
        ss_matches.append({
            "match_id": m_id,
            "home": home,
            "away": away,
            "data": m
        })

    print(f"Loaded {len(imdb_df)} IMDb ratings and {len(ss_matches)} SofaScore matches.")

    rows = []
    matches_matched = 0

    for idx, row in imdb_df.iterrows():
        title = row["title"]
        rating = float(row["rating"])
        votes = row["votes"]
        if pd.isna(votes) or (isinstance(votes, str) and votes.strip() == ""):
            votes = 100 # default votes if missing
        else:
            try:
                # First convert to float to handle float strings like '128.0'
                votes = int(float(str(votes).replace(",", "")))
            except ValueError:
                votes = 100

        # Extract teams from IMDb title e.g. "Group A: Qatar vs. Ecuador" or "Final: Argentina vs. France"
        if ":" not in title:
            print(f"Skipping row with unexpected title format: {title}")
            continue
            
        part = title.split(":", 1)[1].strip()
        if " vs. " not in part:
            print(f"Skipping row with unexpected separator: {title}")
            continue
            
        team_a_raw, team_b_raw = part.split(" vs. ")
        team_a = norm_team(team_a_raw)
        team_b = norm_team(team_b_raw)
        imdb_teams = {team_a, team_b}

        # Find matching SofaScore match
        matched_match = None
        for sm in ss_matches:
            if {sm["home"], sm["away"]} == imdb_teams:
                matched_match = sm
                break

        if matched_match is None:
            print(f"WARNING: Could not find SofaScore match for IMDb teams: {imdb_teams}")
            continue

        matches_matched += 1
        m_id = matched_match["match_id"]
        m_data = matched_match["data"]
        details = m_data.get("details", {}).get("event", {})
        stats = m_data.get("statistics", {})
        incidents = m_data.get("incidents", {}).get("incidents", [])

        # 1. Basic Scores & Winner
        home_score = details.get("homeScore", {}).get("display", 0)
        away_score = details.get("awayScore", {}).get("display", 0)
        winner_code = details.get("winnerCode", 3) # 1=home, 2=away, 3=draw
        
        # 2. Stage mapping
        round_nr = details.get("roundInfo", {}).get("round", 1)
        stage_map = {
            1: "Group stage",
            2: "Group stage",
            3: "Group stage",
            5: "Round of 16",
            27: "Quarterfinals",
            28: "Semifinals",
            50: "Match for 3rd place",
            29: "Final"
        }
        stage = stage_map.get(round_nr, "Group stage")

        # 3. Status Extra Time & Penalty Shootout
        status_desc = details.get("status", {}).get("description", "")
        extra_time = int(status_desc in ("AET", "AP") or round_nr > 3 and any(inc.get("time", 0) > 90 for inc in incidents))
        penalty_shootout = int(status_desc == "AP" or any(inc.get("incidentType") == "penaltyShootout" for inc in incidents))

        # 4. Goals Chronology & Drama features
        goal_incidents = [inc for inc in incidents if inc.get("incidentType") == "goal"]
        # Sort chronologically (lowest time first, then addedTime if present)
        goal_incidents.sort(key=lambda x: (x.get("time", 0), x.get("addedTime") or 0))

        total_goals = home_score + away_score
        goal_difference = abs(home_score - away_score)

        # Calculate lead changes
        lead_changes = 0
        last_leader = None
        for g in goal_incidents:
            h = g.get("homeScore", 0)
            a = g.get("awayScore", 0)
            if h > a:
                curr_leader = "home"
            elif a > h:
                curr_leader = "away"
            else:
                curr_leader = None
            
            if curr_leader is not None:
                if last_leader is not None and curr_leader != last_leader:
                    lead_changes += 1
                last_leader = curr_leader

        # Calculate comeback win/draw
        comeback_win_draw = 0
        if total_goals > 0:
            if home_score == away_score:
                comeback_win_draw = 1 # Draw and score > 0-0 means at least one team was trailing and came back
            elif home_score > away_score:
                # Home won. Check if Home ever trailed
                for g in goal_incidents:
                    if g.get("awayScore", 0) > g.get("homeScore", 0):
                        comeback_win_draw = 1
                        break
            else:
                # Away won. Check if Away ever trailed
                for g in goal_incidents:
                    if g.get("homeScore", 0) > g.get("awayScore", 0):
                        comeback_win_draw = 1
                        break

        # Calculate late goals (scored after min 80 or in extra time)
        late_goals = sum(1 for g in goal_incidents if g.get("time", 0) >= 80)
        time_of_last_goal = max([g.get("time", 0) for g in goal_incidents]) if len(goal_incidents) > 0 else 0

        # 5. Offensive Stats from ALL period
        h_shots, a_shots = get_stat(stats, "totalShotsOnGoal")
        total_shots = h_shots + a_shots

        h_sot, a_sot = get_stat(stats, "shotsOnGoal")
        total_shots_on_target = h_sot + a_sot

        h_xg, a_xg = get_stat(stats, "expectedGoals")
        total_xg = h_xg + a_xg
        xg_difference = abs(h_xg - a_xg)

        h_bc, a_bc = get_stat(stats, "bigChanceCreated")
        big_chances_total = h_bc + a_bc

        h_bcm, a_bcm = get_stat(stats, "bigChanceMissed")
        big_chances_missed = h_bcm + a_bcm

        # 6. Intensity / Flow Stats
        h_pos, a_pos = get_stat(stats, "ballPossession")
        possession_imbalance = abs((h_pos or 50.0) - 50.0)

        h_fouls, a_fouls = get_stat(stats, "fouls")
        fouls_total = h_fouls + a_fouls

        h_corners, a_corners = get_stat(stats, "cornerKicks")
        corner_kicks = h_corners + a_corners

        # 7. Cards & Penalties awarded
        card_incidents = [inc for inc in incidents if inc.get("incidentType") == "card"]
        red_cards = sum(1 for c in card_incidents if c.get("incidentClass") in ("red", "yellowRed"))
        yellow_cards = sum(1 for c in card_incidents if c.get("incidentClass") == "yellow")

        # In-game penalties (scored + missed)
        penalty_goals = sum(1 for g in goal_incidents if g.get("incidentClass") == "penalty")
        missed_penalties = sum(1 for inc in incidents if inc.get("incidentType") == "inGamePenalty" and inc.get("incidentClass") == "missed")
        penalties_awarded = penalty_goals + missed_penalties

        # Append row dict
        rows.append({
            "match_id": m_id,
            "home_team": matched_match["home"],
            "away_team": matched_match["away"],
            "stage": stage,
            "rating": rating,
            "votes": votes,
            "total_goals": total_goals,
            "goal_difference": goal_difference,
            "lead_changes": lead_changes,
            "comeback_win_draw": comeback_win_draw,
            "late_goals": late_goals,
            "time_of_last_goal": time_of_last_goal,
            "extra_time": extra_time,
            "penalty_shootout": penalty_shootout,
            "total_shots": total_shots,
            "total_shots_on_target": total_shots_on_target,
            "total_xg": total_xg,
            "xg_difference": xg_difference,
            "big_chances_total": big_chances_total,
            "big_chances_missed": big_chances_missed,
            "possession_imbalance": possession_imbalance,
            "fouls_total": fouls_total,
            "corner_kicks": corner_kicks,
            "red_cards": red_cards,
            "yellow_cards": yellow_cards,
            "penalties_awarded": penalties_awarded
        })

    # Save to CSV
    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully matched and prepared {matches_matched} out of 64 matches.")
    print(f"Data saved to: {output_path}")

    # Print basic correlation analysis
    print("\n--- Correlation with Enjoyability (IMDb Rating) ---")
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    correlations = final_df[numeric_cols].corr()["rating"].sort_values(ascending=False)
    for col, corr in correlations.items():
        if col != "rating":
            print(f"  - {col:30s}: {corr:6.3f}")

if __name__ == "__main__":
    main()
