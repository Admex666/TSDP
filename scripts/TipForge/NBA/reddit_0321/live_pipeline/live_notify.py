import sys
import os
import pandas as pd
import datetime

pipeline_dir = r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\live_pipeline"
# Append the path to telegram.py
sys.path.append(r"E:\Data\TSDP\scripts\TipForge")

try:
    from telegram import send_to_telegram
except ImportError as e:
    print(f"Error importing telegram module: {e}")
    sys.exit(1)

def generate_live_notify():
    card_path = os.path.join(pipeline_dir, "staging_4_betting_card.csv")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Lépés 5: Notification Integration (Telegram)")
    
    if not os.path.exists(card_path):
        print("❌ Hiba: Nincs Stage 4 fájl (staging_4_betting_card.csv).")
        return
        
    df = pd.read_csv(card_path)
    
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    message = f"🏀 NBA TipForge Betting Card 🏀\n📅 {today_str}\n\n"
    
    if len(df) == 0:
        message += "❌ Mai napra a modell nem talált Value Bet-et (>2% Edge).\nPihentetjük a bankrollt a holnapi meccsekre!"
    else:
        message += f"🎯 Talált Value Betek száma: {len(df)}\n\n"
        
        for idx, bet in df.iterrows():
            message += f"⚔️ {bet['Matchup']}\n"
            message += f"👉 Tipp: {bet['Bet']} (Odds: {bet['Odds']:.2f})\n"
            message += f"📉 Modell Esély: {bet['Model_Prob']*100:.1f}%\n"
            message += f"🔥 Edge (Érték): {bet['Edge']*100:.1f}%\n"
            message += f"💰 Javasolt Tét: {bet['Rec_Units']:.2f}% Bankroll\n"
            message += "--------------------------\n"
            
    message += "\n🤖 Automatikus NBA Pipeline"
    
    print("Üzenet formázva. Küldés Telegramra az 'owner'-nek...")
    send_to_telegram(message, "owner")
    
    print("\n=======================================================")
    print("--- STAGE 5 OUTPUT: TELEGRAM NOTIFICATION SENT ---")
    print("=======================================================")
    print(message)
    print("=======================================================")

if __name__ == "__main__":
    generate_live_notify()
