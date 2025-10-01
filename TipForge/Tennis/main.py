from scrape import get_date_matches
from predictor import predict_tennis_match_simple
from updater import update_tennis_paper, update_betting_results

# NAPI MECCSEK - CSV
date = input("Friss adatok gyűjtéséhez írd be a dátumot (pl. '2025-10-22').\n")
if len(date.split('-') == 3):
    print("🎾 Mai ATP 250+ meccsek gyűjtése és predikálása...")

    match_ids = get_date_matches(date=date, min_points=250)

    print(f"\n📋 Összegyűjtött meccs ID-k ({len(match_ids)} db)")

    if match_ids:
        # DataFrame frissítése
        df = update_tennis_paper(match_ids)
        
        # Összegzés
        print(f"\n📊 ÖSSZEGZÉS:")
        print(f"   - Összes meccs: {len(df)}")
        print(f"   - Ma prediktálva: {len(match_ids)}")
        print(f"   - Erős ajánlások: {len(df[df['best_value'] > 5])}")
        print(f"   - Gyenge ajánlások: {len(df[(df['best_value'] > 2) & (df['best_value'] <= 5)])}")
        
        # Erős ajánlások megjelenítése
        strong_bets = df[df['best_value'] > 5]
        if not strong_bets.empty:
            print(f"\n🔥 ERŐS AJÁNLÁSOK:")
            for _, bet in strong_bets.iterrows():
                print(f"   - {bet['player1_name']} vs {bet['player2_name']}: {bet['bet_placed_on']} ({bet['best_value']:.1f}% value)")
    else:
        print("❌ Nincs ATP 250+ szintű meccs ma!")
else:
    print("❌ Nem dátum formátumú, adatgyűjtés kihagyva.")


# BETTING FRISSÍTÉS
if input("Eredmények frissítéséhez, írd be hogy 'friss'.\n") == "friss":
    print("\n🔄 Fogadási eredmények frissítése...")
    df = update_betting_results()
else:
    print("❌ Frissítés kihagyva.")