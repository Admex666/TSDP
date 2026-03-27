import pandas as pd
import os

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    matches_file = os.path.join(directory, "uel_2025_26_rounds_1-8.csv")
    standings_file = os.path.join(directory, "uel_2025_26_standings.csv")
    
    maccabi_id = 5198
    
    if not os.path.exists(matches_file) or not os.path.exists(standings_file):
        print("Required files not found.")
        return
        
    df_matches = pd.read_csv(matches_file)
    df_standings = pd.read_csv(standings_file)
    
    # Find all opponents of Maccabi
    opponents = []
    for _, row in df_matches.iterrows():
        if row['homeTeam_id'] == maccabi_id:
            opponents.append(row['awayTeam_id'])
        elif row['awayTeam_id'] == maccabi_id:
            opponents.append(row['homeTeam_id'])
            
    # Filter standings for these opponents
    opp_df = df_standings[df_standings['team_id'].isin(opponents)].sort_values(by='pos')
    
    print(f"Maccabi Tel Aviv ellenfelei és végső helyezésük/pontszámuk:")
    print("-" * 60)
    print(opp_df[['pos', 'team_name', 'points', 'won', 'drawn', 'lost']].to_string(index=False))
    print("-" * 60)
    print(f"Átlagos ellenfél pontszám: {opp_df['points'].mean():.2f}")

if __name__ == "__main__":
    main()
