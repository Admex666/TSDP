import pandas as pd

def extract_penalties():
    df = pd.read_csv("magyar_kupa_incidents.csv")
    match_id = 15679214 # Fradi vs ETO
    
    # Filter for penalty shootout
    pens = df[(df['event_id'] == match_id) & (df['type'] == 'penaltyShootout')].copy()
    
    # SofaScore incidents are usually in REVERSE order. Reverse them back to chronological.
    pens = pens.iloc[::-1]
    
    print("--- Büntetőpárbaj sorrendje (Fradi vs ETO) ---")
    for i, (_, row) in enumerate(pens.iterrows()):
        shooter = row['player_name']
        team = "HAZAI (Fradi)" if row['is_home'] else "VENDÉG (ETO)"
        outcome = row['class'] # 'scored' or 'missed'
        print(f"{i+1}. {team}: {shooter} - {outcome}")

if __name__ == "__main__":
    extract_penalties()
