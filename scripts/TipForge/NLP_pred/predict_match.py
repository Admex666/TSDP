import torch
import pandas as pd
from dataset import RisingBallerDataset
from lightning_module import RisingBallerModule
import argparse
import os

def predict(home_team_name, away_team_name, checkpoint_path=None):
    # 1. Beállítások és adatok betöltése
    if not os.path.exists("processed_data.parquet"):
        print("Hiba: processed_data.parquet nem található. Futtasd a preprocess_rolling.py-t!")
        return

    df = pd.read_parquet("processed_data.parquet")
    
    # Kiválasztjuk a játékosok legfrissebb állapotát (utolsó ismert rolling stats)
    # A match_date-et konvertáljuk, ha még nem az
    df['match_date'] = pd.to_datetime(df['game'].str.split(' ').str[0])
    latest_stats = df.sort_values('match_date').groupby('player').last().reset_index()

    # 2. Összeállítjuk a kezdőket (a megadott csapatok legutóbbi meccse alapján)
    home_players = latest_stats[latest_stats['team'] == home_team_name].head(11)
    away_players = latest_stats[latest_stats['team'] == away_team_name].head(11)

    if len(home_players) < 11 or len(away_players) < 11:
        print(f"Hiba: Nem található elég adat az egyik csapathoz.")
        print(f"Tokenek száma - {home_team_name}: {len(home_players)}, {away_team_name}: {len(away_players)}")
        print("Lehetséges, hogy elírtad a csapat nevét (pl. 'Manchester City', 'Arsenal', 'Liverpool').")
        return

    # Dataset példányosítás a vocab-ok és feature nevek eléréséhez
    ds = RisingBallerDataset(data_file="processed_data.parquet")
    
    # Bemeneti tenzorok előkészítése (22 játékos)
    combined = pd.concat([home_players, away_players])
    p_ids = torch.tensor([ds.player_vocab.get(p, 0) for p in combined['player']]).unsqueeze(0)
    pos_ids = torch.tensor([ds.pos_map.get(p, 0) for p in combined['pos_mapped']]).unsqueeze(0)
    feats = torch.tensor(combined[ds.features].fillna(0).values, dtype=torch.float32).unsqueeze(0)

    # 3. Modell betöltése
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Modell betöltése innen: {checkpoint_path}")
        model = RisingBallerModule.load_from_checkpoint(checkpoint_path)
    else:
        print("Figyelem: Nem adtál meg érvényes checkpointot, alapmodellel futok (véletlen súlyok)!")
        model = RisingBallerModule(
            num_players=len(ds.player_vocab) + 1,
            num_positions=len(ds.pos_map) + 1,
            feature_dim=len(ds.features)
        )
    
    model.eval()

    # 4. Inferenciális futtatás
    with torch.no_grad():
        prediction = model({'player_ids': p_ids, 'pos_ids': pos_ids, 'features': feats})
        
    print(f"\n" + "="*30)
    print(f" RISINGBALLER PREDICTION")
    print(f" " + "="*30)
    print(f" Hazai: {home_team_name}")
    print(f" Vendég: {away_team_name}")
    print(f" " + "-"*30)
    print(f" Várható gólok:")
    print(f"   {home_team_name}: {prediction[0][0].item():.2f}")
    print(f"   {away_team_name}: {prediction[0][1].item():.2f}")
    print(f"="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, default="Manchester City", help="Hazai csapat neve")
    parser.add_argument("--away", type=str, default="Arsenal", help="Vendég csapat neve")
    parser.add_argument("--ckpt", type=str, help="Checkpoint (.ckpt) fájl elérési útja")
    args = parser.parse_args()
    
    # Ha nincs megadva checkpoint, megpróbáljuk megkeresni az utolsót
    checkpoint = args.ckpt
    if not checkpoint:
        import glob
        ckpts = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
        if ckpts:
            # A legutolsót választjuk (időrendben)
            checkpoint = max(ckpts, key=os.path.getmtime)
    
    predict(args.home, args.away, checkpoint)
