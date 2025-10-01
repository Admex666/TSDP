import pandas as pd
import os

def get_betting_statistics(csv_path='Tennis/tennis_paper.csv'):
    """
    Összesített fogadási statisztikák lekérése a CSV-ből
    """
    if not os.path.exists(csv_path):
        print("❌ CSV fájl nem található!")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Hiba a CSV betöltésekor: {e}")
        return None
    
    # Csak a lezárt fogadások
    settled_bets = df[df['bet_settled'] == True]
    
    if settled_bets.empty:
        print("ℹ️  Nincs lezárt fogadás a statisztikákhoz")
        return None
    
    # Alap statisztikák
    total_bets = len(settled_bets)
    won_bets = len(settled_bets[settled_bets['result'] == 'WON'])
    lost_bets = len(settled_bets[settled_bets['result'] == 'LOST'])
    
    # Profit számítások
    total_profit = settled_bets['profit'].sum()
    total_stake = settled_bets['bet_stake_percent'].sum()
    
    # Átlagok
    avg_odds_p1 = settled_bets['player1_odds'].mean()
    avg_odds_p2 = settled_bets['player2_odds'].mean()
    avg_stake = settled_bets['bet_stake_percent'].mean()
    avg_prob = settled_bets[['player1_pred_prob', 'player2_pred_prob']].max(axis=1).mean()
    
    # Value alapú bontás
    strong_bets = settled_bets[settled_bets['best_value'] > 5]
    weak_bets = settled_bets[(settled_bets['best_value'] > 2) & (settled_bets['best_value'] <= 5)]
    
    # Surface bontás
    surface_stats = settled_bets.groupby('surface').agg({
        'result': lambda x: (x == 'WON').sum() / len(x) * 100,
        'profit': 'sum',
        'bet_stake_percent': 'sum'
    }).round(2)
    
    # Tournament points bontás
    tournament_stats = settled_bets.groupby('tournament').agg({
        'result': lambda x: (x == 'WON').sum() / len(x) * 100,
        'profit': 'sum',
        'bet_stake_percent': 'sum'
    }).round(2)
    
    # Hónap bontás (ha van dátum)
    if 'date' in df.columns:
        try:
            settled_bets['month'] = pd.to_datetime(settled_bets['date']).dt.to_period('M')
            monthly_stats = settled_bets.groupby('month').agg({
                'result': lambda x: (x == 'WON').sum(),
                'profit': 'sum',
                'bet_stake_percent': 'sum',
                'event_id': 'count'
            }).round(2)
            monthly_stats['hit_rate'] = (monthly_stats['result'] / monthly_stats['event_id'] * 100).round(1)
        except:
            monthly_stats = None
    else:
        monthly_stats = None
    
    # Statisztikák megjelenítése
    print("\n📊 ÖSSZESÍTETT FOGADÁSI STATISZTIKÁK")
    print("=" * 60)
    
    # Alap statisztikák
    hit_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
    
    print(f"🎯 ÖSSZES FOGADÁS: {total_bets}")
    print(f"✅ Nyert fogadások: {won_bets} ({hit_rate:.1f}%)")
    print(f"❌ Vesztett fogadások: {lost_bets} ({100-hit_rate:.1f}%)")
    print(f"💰 Összes profit: {total_profit:+.2f}%")
    print(f"📈 ROI: {roi:+.1f}%")
    print(f"💵 Összes tét: {total_stake:.1f}%")
    print(f"📊 Átlagos tét: {avg_stake:.2f}%")
    print(f"🎲 Átlagos odds: {avg_odds_p1:.2f} (P1) / {avg_odds_p2:.2f} (P2)")
    print(f"🔮 Átlagos predikciós valószínűség: {avg_prob:.1%}")
    
    # Value bontás
    print(f"\n💎 VALUE BONTÁS:")
    if not strong_bets.empty:
        strong_hit_rate = (len(strong_bets[strong_bets['result'] == 'WON']) / len(strong_bets) * 100)
        strong_profit = strong_bets['profit'].sum()
        print(f"   🔥 Erős value (>5%): {len(strong_bets)} fogadás, {strong_hit_rate:.1f}% találat, {strong_profit:+.2f}% profit")
    
    if not weak_bets.empty:
        weak_hit_rate = (len(weak_bets[weak_bets['result'] == 'WON']) / len(weak_bets) * 100)
        weak_profit = weak_bets['profit'].sum()
        print(f"   ⚡ Gyenge value (2-5%): {len(weak_bets)} fogadás, {weak_hit_rate:.1f}% találat, {weak_profit:+.2f}% profit")
    
    # Felület bontás
    print(f"\n🎾 FELÜLET BONTÁS:")
    for surface, stats in surface_stats.iterrows():
        hit_rate = stats['result']
        profit = stats['profit']
        print(f"   {surface}: {hit_rate:.1f}% találat, {profit:+.2f}% profit")
    
    # Tornaszint bontás
    print(f"\n🏆 TORNA BONTÁS:")
    for tour, stats in tournament_stats.iterrows():
        hit_rate = stats['result']
        profit = stats['profit']
        print(f"   {tour}: {hit_rate:.1f}% találat, {profit:+.2f}% profit")
    
    # Havi bontás (ha elérhető)
    if monthly_stats is not None and not monthly_stats.empty:
        print(f"\n📅 HAVI BONTÁS:")
        for month, stats in monthly_stats.iterrows():
            hit_rate = stats['hit_rate']
            profit = stats['profit']
            count = stats['event_id']
            print(f"   {month}: {count} fogadás, {hit_rate:.1f}% találat, {profit:+.2f}% profit")
    
    # Teljesítmény értékelés
    print(f"\n🏆 TELJESÍTMÉNY ÉRTÉKELÉS:")
    if roi > 20:
        rating = "KIVÁLÓ 🔥"
    elif roi > 10:
        rating = "NAGYON JÓ ✅" 
    elif roi > 0:
        rating = "JOBB, MINT A VÁRHATÓ 📈"
    elif roi > -5:
        rating = "ÁTLAGOS ⚖️"
    elif roi > -10:
        rating = "GYENGE 📉"
    else:
        rating = "ROSSZ ❌"
    
    print(f"   Értékelés: {rating}")
    print(f"   Összesített eredmény: {'Profit' if total_profit > 0 else 'Veszteség'}")
    
    print("=" * 60)
    
    return {
        'total_bets': total_bets,
        'won_bets': won_bets,
        'lost_bets': lost_bets,
        'hit_rate': hit_rate,
        'total_profit': total_profit,
        'roi': roi,
        'total_stake': total_stake,
        'avg_stake': avg_stake,
        'strong_bets_count': len(strong_bets),
        'weak_bets_count': len(weak_bets)
    }