import pandas as pd

# Explicit lists provided by the USER
NB1_TEAMS = [
    "ETO FC Győr", "Ferencváros TC", "Debreceni VSC", "Zalaegerszegi TE", 
    "Paksi FC", "Újpest", "Kisvárda FC", "Puskás Akadémia", 
    "MTK Budapest", "Nyiregyháza Spartacus", "Diósgyőri VTK", "Kazincbarcikai SC"
]

NB2_TEAMS = [
    "Vasas", "Budapest Honvéd FC", "Kecskemét TE", "Mezőkövesd Zsóry", 
    "Aqvital FC Csákvár", "Videoton FC Fehérvár", "Kozármisleny FC", 
    "BVSC-Zugló", "Karcagi SC", "Tiszakécskei LC", "FC Ajka", 
    "Szeged-Csanád Grosics Akadémia", "Soroksár SC", "Békéscsaba 1912 Előre", 
    "Budafoki MTE", "Szentlőrinc SE"
]

def get_tier_label(team_name, leagues):
    # Check explicit NB I list
    if any(nb1 in team_name for nb1 in NB1_TEAMS):
        return "NB I"
        
    # Check explicit NB II list
    if any(nb2 in team_name for nb2 in NB2_TEAMS):
        return "NB II"
        
    if not isinstance(leagues, str): return "Ismeretlen"
    leagues = leagues.lower()
    
    # NB III
    if 'nb iii' in leagues or 'nbiii' in leagues:
        return "NB III"
        
    # County / BLSZ
    if any(x in leagues for x in ['oszt', 'blsz', 'megye', 'területi']):
        return "Megye"
        
    return "Egyéb/Amatőr"

def get_tier_rank(tier):
    ranks = {"NB I": 1, "NB II": 2, "NB III": 3, "Megye": 4, "Egyéb/Amatőr": 5, "Ismeretlen": 6}
    return ranks.get(tier, 99)

def run_analysis():
    # 1. Load Data
    matches = pd.read_csv("magyar_kupa_matches_sofascore.csv")
    teams = pd.read_csv("magyar_kupa_teams_leagues.csv")
    
    # Create mapping
    team_tiers = {row['team_name']: get_tier_label(row['team_name'], row['leagues']) for _, row in teams.iterrows()}
    
    # Enrich matches with tiers
    matches['home_tier'] = matches['home_team'].map(team_tiers)
    matches['away_tier'] = matches['away_team'].map(team_tiers)
    
    print(f"Loaded {len(matches)} matches and {len(teams)} teams.")
    
    # 2. Round 3 Distribution
    r3 = matches[matches['round'] == 'Round 3'].copy()
    print(f"Found {len(r3)} matches in Round 3.")
    
    all_r3_teams = pd.concat([r3['home_team'], r3['away_team']])
    r3_tiers = all_r3_teams.map(team_tiers).value_counts()
    
    print("--- Round 3 Csapatok Megoszlása ---")
    print(r3_tiers.to_string())
    
    # 3. Upsets in Round 3
    # An upset is when a lower tier team (higher rank number) beats a higher tier team (lower rank number)
    def is_upset(row):
        h_rank = get_tier_rank(row['home_tier'])
        a_rank = get_tier_rank(row['away_tier'])
        
        # Determine winner
        # Note: score_home and score_away are final display scores
        if row['score_home'] > row['score_away']:
            # Home won. Was it an upset?
            return h_rank > a_rank
        elif row['score_away'] > row['score_home']:
            # Away won. Was it an upset?
            return a_rank > h_rank
        else:
            # Draw (shouldn't happen in Cup results unless we don't have the winner)
            return False

    upsets = r3[r3.apply(is_upset, axis=1)]
    print("\n--- Upsets a Round 3-ban ---")
    for _, row in upsets.iterrows():
        winner = row['home_team'] if row['score_home'] > row['score_away'] else row['away_team']
        loser = row['away_team'] if row['score_home'] > row['score_away'] else row['home_team']
        w_tier = team_tiers.get(winner)
        l_tier = team_tiers.get(loser)
        print(f"Bravúr: {winner} ({w_tier}) kiejtette: {loser} ({l_tier}) - Eredmény: {row['score_home']}:{row['score_away']}")

    # 4. NB II vs NB I matches
    # Find matches where one team is NB I and other is NB II
    nb1_vs_nb2 = matches[((matches['home_tier'] == 'NB I') & (matches['away_tier'] == 'NB II')) | 
                         ((matches['home_tier'] == 'NB II') & (matches['away_tier'] == 'NB I'))].copy()
    
    print("\n--- NB II vs NB I mérkőzések Részletes Statisztikája ---")
    print(f"Összesen: {len(nb1_vs_nb2)} meccs")
    
    nb2_w, nb2_d, nb2_l = 0, 0, 0
    nb2_gf, nb2_ga = 0, 0
    nb2_progressed = 0
    
    # Load incidents to resolve draws
    inc_df = pd.read_csv("magyar_kupa_incidents.csv")
    
    for _, row in nb1_vs_nb2.iterrows():
        h_score, a_score = row['score_home'], row['score_away']
        eid = row['event_id']
        
        if row['home_tier'] == 'NB II':
            gf, ga = h_score, a_score
        else:
            gf, ga = a_score, h_score
            
        nb2_gf += gf
        nb2_ga += ga
        
        if h_score > a_score:
            winner_tier = row['home_tier']
            if winner_tier == 'NB II': nb2_progressed += 1
        elif a_score > h_score:
            winner_tier = row['away_tier']
            if winner_tier == 'NB II': nb2_progressed += 1
        else:
            winner_tier = "Döntetlen"
            # Resolve draw via incidents
            match_inc = inc_df[inc_df['event_id'] == eid]
            pens = match_inc[match_inc['type'] == 'penaltyShootout']
            if not pens.empty:
                # The last penalty shootout incident has the final shootout score
                last_pen = pens.iloc[0] # Incidents are reverse chronological
                if last_pen['home_score'] > last_pen['away_score']:
                    advancing_team_tier = row['home_tier']
                else:
                    advancing_team_tier = row['away_tier']
                
                if advancing_team_tier == 'NB II':
                    nb2_progressed += 1
            
        if winner_tier == 'NB II':
            nb2_w += 1
        elif winner_tier == 'NB I':
            nb2_l += 1
        else:
            nb2_d += 1
            
    print(f"NB II eredmények (90/120p): {nb2_w} GY, {nb2_d} D, {nb2_l} V")
    print(f"NB II továbbjutások: {nb2_progressed} alkalom ({(nb2_progressed/len(nb1_vs_nb2))*100:.1f}%)")
    print(f"NB II gólkülönbség: {nb2_gf} - {nb2_ga} ({nb2_gf - nb2_ga:+.0f})")
    if len(nb1_vs_nb2) > 0:
        print(f"Átlagos NB II szerzett gól: {nb2_gf/len(nb1_vs_nb2):.2f}")
        print(f"Átlagos NB II kapott gól: {nb2_ga/len(nb1_vs_nb2):.2f}")

    # 5. Semifinalists' Path
    semifinalists = ['Ferencváros TC', 'ETO FC Győr', 'Budapest Honvéd FC', 'Zalaegerszegi TE']
    
    print("\n--- Elődöntősök útja ---")
    for semi in semifinalists:
        print(f"\n>> {semi.upper()} útvonala:")
        # Find all matches involving this team
        path = matches[(matches['home_team'] == semi) | (matches['away_team'] == semi)].copy()
        # Sort by round if possible (we'll just use the order in CSV for now which is roughly chronological)
        
        for _, row in path.iterrows():
            opponent = row['away_team'] if row['home_team'] == semi else row['home_team']
            opp_tier = team_tiers.get(opponent, "Ismeretlen")
            res = f"{row['score_home']}:{row['score_away']}"
            print(f"  - {row['round']}: vs {opponent} ({opp_tier}) -> {res}")

if __name__ == "__main__":
    run_analysis()
