import sqlite3
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SorareModelAnalytics:
    def __init__(self, db_path="sorare_historical.db"):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_data(self):
        """Betölti az összes szükséges adatot a számításokhoz."""
        conn = self.get_connection()
        
        df_players = pd.read_sql_query("SELECT * FROM players", conn)
        df_matches = pd.read_sql_query("SELECT * FROM match_performances", conn)
        df_auctions = pd.read_sql_query("SELECT * FROM auctions", conn)
        
        conn.close()
        return df_players, df_matches, df_auctions

    def run_analysis(self):
        """
        Kiszámítja az összes fejlett feature-t (statisztikák, ár/teljesítmény arányok),
        majd kiértékeli a 3 modellt/stratégiát a piaci rések megtalálására.
        """
        df_players, df_matches, df_auctions = self.load_data()
        
        if df_players.empty:
            logger.warning("Nincs játékos adat az elemzéshez.")
            return [], []

        # 1. FEJLETT MECCS STATISZTIKÁK SZÁMÍTÁSA JÁTÉKOSONKÉNT
        player_features = []
        
        for idx, player in df_players.iterrows():
            p_id = player['id']
            p_name = player['display_name']
            
            # Kiszűrjük a játékos meccseit
            p_matches = df_matches[df_matches['player_id'] == p_id].sort_values(by='match_date', ascending=False)
            
            # Áradatok kiszűrése
            p_auctions = df_auctions[df_auctions['player_id'] == p_id]
            listings = p_auctions[p_auctions['price_type'] == 'direct_listing']
            sales = p_auctions[p_auctions['price_type'] == 'recent_sale']
            
            # Árak meghatározása
            floor_price = listings['price_eur'].min() if not listings.empty else None
            avg_sale_price = sales['price_eur'].mean() if not sales.empty else None
            
            # Meccs statisztikák
            total_matches = len(p_matches)
            
            if total_matches > 0:
                scores = p_matches['total_score'].values
                decisives = p_matches['decisive_score'].values
                all_arounds = p_matches['all_around_score'].values
                
                l5_score = round(float(np.mean(scores[:5])), 2) if total_matches >= 5 else round(float(np.mean(scores)), 2)
                l15_score = round(float(np.mean(scores[:15])), 2) if total_matches >= 15 else round(float(np.mean(scores)), 2)
                
                # Kiszámoljuk a szórást (volatilitás - rizikófaktor)
                std_score = round(float(np.std(scores)), 2) if total_matches > 1 else 0.0
                
                # Decisive vs All-Around arány
                avg_decisive = float(np.mean(decisives))
                avg_all_around = float(np.mean(all_arounds))
                total_avg = float(np.mean(scores))
                
                decisive_share = round((avg_decisive / total_avg) * 100, 1) if total_avg > 0 else 0.0
                all_around_share = round((avg_all_around / total_avg) * 100, 1) if total_avg > 0 else 0.0
                
                # Hazai vs Idegenbeli átlagok
                home_matches = p_matches[p_matches['is_home'] == 1]
                away_matches = p_matches[p_matches['is_home'] == 0]
                
                home_avg = round(float(home_matches['total_score'].mean()), 2) if not home_matches.empty else None
                away_avg = round(float(away_matches['total_score'].mean()), 2) if not away_matches.empty else None
            else:
                # Ha nincsenek konkrét meccsek, a DB átlagból indulunk ki
                l5_score = player['average_score']
                l15_score = player['average_score']
                std_score = 0.0
                decisive_share = 0.0
                all_around_share = 0.0
                home_avg = None
                away_avg = None
            
            # Ár/Teljesítmény mutató (EUR / L15 pont)
            price_to_perf = round(floor_price / l15_score, 3) if floor_price and l15_score and l15_score > 0 else None
            
            # Piaci diszkont (historikus eladási átlag vs floor ár)
            discount = round(((avg_sale_price - floor_price) / avg_sale_price) * 100, 1) if floor_price and avg_sale_price and avg_sale_price > 0 else 0.0
            
            features = {
                'id': p_id,
                'display_name': p_name,
                'position': player['position'],
                'age': player['age'],
                'club_name': player['club_name'],
                'is_injured': int(player.get('is_injured', 0)),
                'is_suspended': int(player.get('is_suspended', 0)),
                'l5_score': l5_score,
                'l15_score': l15_score,
                'std_score': std_score,
                'decisive_share': decisive_share,
                'all_around_share': all_around_share,
                'home_avg_score': home_avg,
                'away_avg_score': away_avg,
                'floor_price': floor_price,
                'avg_sale_price': round(avg_sale_price, 2) if avg_sale_price else None,
                'discount_percent': discount,
                'price_to_performance': price_to_perf
            }
            player_features.append(features)
            
        df_features = pd.DataFrame(player_features)
        
        # 2. PIACI RÉSEK AZONOSÍTÁSA (ALERTEK GENERÁLÁSA)
        alerts = []
        
        for idx, row in df_features.iterrows():
            floor = row['floor_price']
            avg_sale = row['avg_sale_price']
            l15 = row['l15_score']
            l5 = row['l5_score']
            std = row['std_score']
            p_to_p = row['price_to_performance']
            disc = row['discount_percent']
            p_name = row['display_name']
            
            if not floor:
                continue
                
            # Stratégia 1: Buy the Dip (Vedd meg a hullámvölgyet!)
            # Feltételek: > 20% diszkont, jó pontszám, stabil játékos
            if disc >= 20.0 and l15 >= 45.0:
                alerts.append({
                    'player_name': p_name,
                    'strategy': 'Buy the Dip',
                    'urgency': 'HIGH' if disc >= 30.0 else 'MEDIUM',
                    'metric': f"Diszkont: {disc}%",
                    'details': f"A játékos jelenlegi legolcsóbb piaci ára ({floor} EUR) szignifikánsan elmarad a 30 napos historikus átlagártól ({avg_sale} EUR). Kiváló alkalmi vétel visszatérésre vagy továbbértékesítésre!"
                })
                
            # Stratégia 2: Undervalued Utility (Alulértékelt Pontgyárosok)
            # Feltételek: Olcsó pontszám (EUR/L15 < 0.35), jó átlag (>48), stabil teljesítmény (std < 15)
            if p_to_p and p_to_p < 0.35 and l15 >= 48.0 and std < 15.0:
                alerts.append({
                    'player_name': p_name,
                    'strategy': 'Undervalued Utility',
                    'urgency': 'HIGH' if p_to_p < 0.25 else 'MEDIUM',
                    'metric': f"EUR/Pont: {p_to_p} EUR",
                    'details': f"Rendkívül konzisztens mezőnyjátékos (Alacsony szórás: {std} pont). Minden egyes L15 pontja mindössze {p_to_p} EUR-ba kerül, ami messze elmarad a piac {round(df_features['price_to_performance'].dropna().mean(), 2) if not df_features['price_to_performance'].dropna().empty else 0.45} EUR/pontos átlagától!"
                })
                
            # Stratégia 3: Form Breakout (Forma-arbitrázs)
            # Feltételek: L5 lényegesen jobb mint az L15 (> 6 pont javulás), és még mindig jó áron van
            if (l5 - l15) >= 6.0 and p_to_p and p_to_p < 0.60:
                alerts.append({
                    'player_name': p_name,
                    'strategy': 'Form Breakout',
                    'urgency': 'MEDIUM',
                    'metric': f"Forma javulás: +{round(l5 - l15, 1)} pont",
                    'details': f"A játékos kiemelkedő formát mutat az utolsó meccsein (L5: {l5} pont vs L15: {l15} pont). A piac még nem árazta be teljesen a formajavulását, a kártyája még mindig kedvező áron szerezhető meg!"
                })
                
        return df_features, alerts

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    analytics = SorareModelAnalytics()
    df_feat, alerts = analytics.run_analysis()
    
    print("\n--- JÁTÉKOS FEATURE MÁTRIX ---")
    columns_to_show = ['display_name', 'position', 'l5_score', 'l15_score', 'std_score', 'floor_price', 'avg_sale_price', 'discount_percent', 'price_to_performance']
    print(df_feat[columns_to_show].to_string(index=False))
    
    print("\n--- DETEKTÁLT PIACI RÉSEK (ALERTEK) ---")
    for a in alerts:
        print(f"[{a['strategy'].upper()} - {a['urgency']}] {a['player_name']} -> {a['metric']}")
        print(f"   Részletek: {a['details']}\n")
