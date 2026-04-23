import pandas as pd
import os
import sys

# Add directory to path to import from simulate_penalties
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulate_penalties import get_win_prob

def export_to_excel():
    df_incidents = pd.read_csv("magyar_kupa_incidents.csv")
    df_matches = pd.read_csv("magyar_kupa_matches_sofascore.csv")
    
    # Target matches
    match_ids = [15679214, 15679215]
    all_rows = []
    
    for match_id in match_ids:
        # Get team names
        match_row = df_matches[df_matches['event_id'] == match_id]
        if match_row.empty: continue
        
        home_name = match_row.iloc[0]['home_team']
        away_name = match_row.iloc[0]['away_team']
        match_label = f"{home_name} vs {away_name}"
        
        # Get penalties
        pens = df_incidents[(df_incidents['event_id'] == match_id) & (df_incidents['type'] == 'penaltyShootout')].copy()
        pens = pens.iloc[::-1] # Chronological
        
        # Initial state
        all_rows.append({
            'Párosítás': match_label,
            'Tizenegyes köre': 0,
            'Játékos neve': 'Kezdés',
            'Csapat': '-',
            'Esély (Home %)': round(get_win_prob(0, 0, 0, 1) * 100, 2)
        })
        
        s1, s2 = 0, 0
        for i, (_, row) in enumerate(pens.iterrows()):
            outcome = row['class']
            is_home = row['is_home'] == True
            player = row['player_name']
            
            if outcome == "scored":
                if is_home: s1 += 1
                else: s2 += 1
            
            next_turn = 1 if is_home else 0
            next_rnd = (i // 2) + 1 + (0 if is_home else 1)
            
            prob = get_win_prob(s1, s2, next_turn, next_rnd)
            
            all_rows.append({
                'Párosítás': match_label,
                'Tizenegyes köre': i + 1,
                'Játékos neve': player,
                'Csapat': home_name if is_home else away_name,
                'Esély (Home %)': round(prob * 100, 2)
            })

    # Save to Excel
    df_final = pd.DataFrame(all_rows)
    output_file = "magyar_kupa_shootouts.xlsx"
    
    try:
        # Using openpyxl as engine
        df_final.to_excel(output_file, index=False)
        print(f"Sikeresen mentve: {output_file}")
    except Exception as e:
        print(f"Hiba az Excel mentésekor: {e}")
        # Fallback to CSV if Excel fails
        csv_file = "magyar_kupa_shootouts.csv"
        df_final.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Fallback: Mentve CSV-be: {csv_file}")

if __name__ == "__main__":
    export_to_excel()
