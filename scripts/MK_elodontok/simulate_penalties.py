import pandas as pd
from functools import lru_cache

# Probability of scoring a penalty
P_SCORE = 0.75

@lru_cache(None)
def get_win_prob(s1, s2, turn, rnd):
    """
    Calculates probability of Team 1 (Fradi) winning the shootout.
    s1, s2: current scores
    turn: 0 if Team 1 is next, 1 if Team 2 is next
    rnd: current round (1 to 5)
    """
    # Remaining shots for each team in the regular 5 rounds
    rem1 = 5 - (rnd - 1) if turn == 0 else 5 - rnd
    rem2 = 5 - (rnd - 1)
    
    # Check if a team has already won (cannot be caught)
    if s1 > s2 + rem2:
        return 1.0
    if s2 > s1 + rem1:
        return 0.0
    
    # If all 5 rounds are completed
    if rnd > 5:
        if s1 > s2: return 1.0
        if s2 > s1: return 0.0
        # Sudden death: both teams have equal p, so 50-50
        return 0.5
    
    # Recursive step
    if turn == 0:
        # Team 1 (Fradi) shoots
        return P_SCORE * get_win_prob(s1 + 1, s2, 1, rnd) + (1 - P_SCORE) * get_win_prob(s1, s2, 1, rnd)
    else:
        # Team 2 (ETO) shoots
        return P_SCORE * get_win_prob(s1, s2 + 1, 0, rnd + 1) + (1 - P_SCORE) * get_win_prob(s1, s2, 0, rnd + 1)

def run_simulation(match_id=15679214):
    # Load sequence from CSV
    df = pd.read_csv("magyar_kupa_incidents.csv")
    pens = df[(df['event_id'] == match_id) & (df['type'] == 'penaltyShootout')].copy()
    
    if pens.empty:
        print(f"Nincs büntetőpárbaj adat a következőhöz: {match_id}")
        return

    # Chronological order
    pens = pens.iloc[::-1]
    
    sequence = []
    for _, row in pens.iterrows():
        sequence.append((row['player_name'], row['is_home'] == True, row['class']))

    import matplotlib.pyplot as plt

    events = ["Kezdés"]
    probs = [get_win_prob(0, 0, 0, 1)]

    s1, s2 = 0, 0
    for i, (player, is_home, outcome) in enumerate(sequence):
        if outcome == "scored":
            if is_home: s1 += 1
            else: s2 += 1
        
        next_turn = 1 if is_home else 0
        next_rnd = (i // 2) + 1 + (0 if is_home else 1)
        
        p = get_win_prob(s1, s2, next_turn, next_rnd)
        
        # Format label: Name (S/M)
        outcome_str = "berúgta" if outcome == "scored" else "kihagyta"
        label = f"{player} ({outcome_str})"
        events.append(label)
        probs.append(p)

    # Load team names for title
    matches_df = pd.read_csv("magyar_kupa_matches_sofascore.csv")
    match_row = matches_df[matches_df['event_id'] == match_id]
    if not match_row.empty:
        home_name = match_row.iloc[0]['home_team']
        away_name = match_row.iloc[0]['away_team']
    else:
        home_name, away_name = "Hazai", "Vendég"
    # Print Table (as requested)
    print(f"\n--- Büntetőpárbaj szimuláció: {home_name} vs {away_name} ---")
    print(f"{'Esemény':<40} | {'Valószínűség (' + home_name + ')':<20}")
    print("-" * 65)
    
    for event, prob in zip(events, probs):
        print(f"{event:<40} | {prob*100:>10.1f}%")

    import matplotlib.pyplot as plt
    # Plotting
    plt.figure(figsize=(12, 6), dpi=120)
    plt.style.use('ggplot')
    
    # Convert to percentages
    probs_pct = [p * 100 for p in probs]
    
    # Line plot with markers
    plt.plot(range(len(events)), probs_pct, marker='o', linestyle='-', linewidth=3, color='#006B3D', markersize=8, label=f'{home_name} továbbjutási esély')
    
    # Fill under the curve
    plt.fill_between(range(len(events)), probs_pct, alpha=0.1, color='#006B3D')
    
    # 50% threshold line
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Egyenlő)')
    
    # Customizing axes
    plt.xticks(range(len(events)), events, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.ylabel('Továbbjutási esély (%)', fontsize=12, fontweight='bold')
    plt.title(f'Büntetőpárbaj valószínűség: {home_name} vs {away_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Adding data labels
    for i, p in enumerate(probs_pct):
        plt.annotate(f"{p:.1f}%", (i, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Excel Export
    excel_data = []
    # Add initial state
    excel_data.append({
        'csapat': 'Kezdés',
        'tizenegyes köre': 0,
        'játékos neve': '-',
        'esély': probs[0]
    })
    
    # Add each penalty
    # Re-extract team names for each shot to be accurate
    s1, s2 = 0, 0
    for i, (player, is_home, outcome) in enumerate(sequence):
        # We need the prob AFTER this shot, which is in probs[i+1]
        team_shooting = home_name if is_home else away_name
        excel_data.append({
            'csapat': team_shooting,
            'tizenegyes köre': i + 1,
            'játékos neve': player,
            'esély': probs[i+1]
        })
    
    df_excel = pd.DataFrame(excel_data)
    excel_file = f"penalty_probs_{match_id}.xlsx"
    try:
        df_excel.to_excel(excel_file, index=False)
        print(f"Excel mentve: {excel_file}")
    except Exception as e:
        # Fallback to CSV if openpyxl is missing
        csv_fallback = excel_file.replace(".xlsx", ".csv")
        df_excel.to_csv(csv_fallback, index=False, encoding='utf-8-sig')
        print(f"Excel mentés sikertelen (hiányzó engine?), CSV mentve: {csv_fallback}")

if __name__ == "__main__":
    import sys
    mid = 15679214 # Default
    if len(sys.argv) > 1:
        try:
            mid = int(sys.argv[1])
        except:
            pass
    run_simulation(mid)
