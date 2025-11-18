from nba_api_module import create_ml_row
import pandas as pd
import os
import time
import random

CSV_PATH = 'data/ml_ready_2024_25.csv'

def safe_call(fn, max_retries=2):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            wait = 2 + attempt*1
            print(f"   → Error: {e} | retry {attempt+1}/{max_retries} after {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

def update_training_csv(game_ids):

    # MÁR LÉTEZŐ CSV BEOLVASÁSA
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        existing_ids = df_existing["game_id"].unique().tolist()
        file_exists = True
    else:
        existing_ids = []
        file_exists = False

    total = len(game_ids)

    # FŐ LOOP
    for i, gid in enumerate(game_ids, start=1):
        gid = str(gid)
        gid = gid if gid.startswith("00") else f"00{gid}"

        print(f"[{i}/{total}] Processing game_id: {gid}")

        if (gid in existing_ids) or (int(gid) in existing_ids):
            print(f"   → Skip: already in CSV")
            continue

        # --- ÚJ SOR KÉSZÍTÉSE ---
        row = safe_call(lambda: create_ml_row(gid))

        # --- CSV-BE ÍRÁS MINDEN SOR UTÁN ---
        # DataFrame-be csomagoljuk, hogy appendelhető legyen
        df_row = pd.DataFrame([row])

        df_row.to_csv(
            CSV_PATH,
            mode='a' if file_exists else 'w',
            header=not file_exists,
            index=False
        )
        file_exists = True  # innentől kezdve append

        existing_ids.append(gid)  # ne dolgozzuk fel újra, ha duplán is lenne bemeneten

        print(f"   → Saved to CSV")
        #  rate limit
        time.sleep(random.uniform(1.0, 2.0))

    print("Update finished.")


games = pd.read_csv('data/games_2024_25.csv')
game_ids = games['GAME_ID'].unique()

update_training_csv(game_ids)
