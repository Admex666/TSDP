
import pandas as pd
import numpy as np

def run_analysis():
    # Adatok betöltése
    df_details = pd.read_csv('match_details_09_26.csv')
    df_odds = pd.read_csv('match_odds_09_26.csv')
    df_incidents = pd.read_csv('match_incidents_09_26.csv')

    # Szűrés a 15/16-os szezontól kezdődően
    def get_season_year(s):
        try:
            year_str = s.split(' ')[-1].split('/')[0]
            return int(year_str)
        except:
            return 0

    df_details = df_details[df_details['season'].apply(get_season_year) >= 15].copy()
    valid_match_ids = df_details['match_id'].unique()
    
    df_odds = df_odds[df_odds['match_id'].isin(valid_match_ids)].copy()
    df_incidents = df_incidents[df_incidents['match_id'].isin(valid_match_ids)].copy()

    # Események idejének előszámítása
    df_incidents['total_time'] = df_incidents['time'] + df_incidents['added_time'].fillna(0)

    print(f"--- Magyar Kupa Döntők Mélyelemzése (2015/16 - 2025/26) ---\n")
    print("MEGJEGYZÉS: Favoritnak CSAK a 66% feletti kupagyőzelmi eséllyel rendelkező csapatokat tekintjük.\n")

    # 1. Döntős részvételek száma
    teams = pd.concat([df_details['home_team'], df_details['away_team']])
    participation = teams.value_counts()
    print("--- 1. Döntős részvételek száma (csökkenő) ---")
    for team, count in participation.items():
        print(f"{team}: {count}")
    print("\n")

    # 2. Mérkőzések eldőlésének módja
    print("--- 2. Mérkőzések eldőlésének módja ---")
    for _, row in df_details.iterrows():
        res = row['status']
        res_hu = "Rendes játékidő" if res == "Ended" else res
        print(f"{row['season']}: {row['home_team']} {row['home_score']}-{row['away_score']} {row['away_team']} -> {res_hu}")
    print("\n")

    # 3. Odds elemzés és Favorit meghatározás (>66%)
    print("--- 3. Odds elemzés és a favorit (esély > 66%) teljesítménye ---")
    odds_summary = []
    for match_id in valid_match_ids:
        match_odds = df_odds[df_odds['match_id'] == match_id]
        if match_odds.empty: continue
            
        details = df_details[df_details['match_id'] == match_id].iloc[0]
        h = match_odds[match_odds['name'] == '1']['odds'].values[0] if not match_odds[match_odds['name'] == '1'].empty else None
        d = match_odds[match_odds['name'] == 'X']['odds'].values[0] if not match_odds[match_odds['name'] == 'X'].empty else None
        a = match_odds[match_odds['name'] == '2']['odds'].values[0] if not match_odds[match_odds['name'] == '2'].empty else None
        
        if h is None or d is None or a is None: continue
            
        # Implikált valószínűségek (margin nélkül)
        inv_h, inv_d, inv_a = 1/h, 1/d, 1/a
        sum_inv = inv_h + inv_d + inv_a
        p_h, p_d, p_a = inv_h/sum_inv, inv_d/sum_inv, inv_a/sum_inv
        
        p_home_cup = p_h + 0.5 * p_d
        p_away_cup = p_a + 0.5 * p_d
        
        favorite = None
        p_fav = 0
        if p_home_cup > 0.66:
            favorite = "Hazai"
            p_fav = p_home_cup
        elif p_away_cup > 0.66:
            favorite = "Vendég"
            p_fav = p_away_cup
            
        if favorite:
            winner_code = details['winner_code']
            favorite_won = (favorite == "Hazai" and winner_code == 1) or (favorite == "Vendég" and winner_code == 2)
            
            odds_summary.append({
                'match_id': match_id,
                'favorite': favorite,
                'p_fav': p_fav,
                'favorite_won': favorite_won,
                'home_team': details['home_team'],
                'away_team': details['away_team'],
                'season': details['season'],
                'diff': abs(h - a)
            })
            won_str = "Igen" if favorite_won else "Nem"
            print(f"{details['season']}: Favorit: {favorite} (Esély: {p_fav*100:.1f}%), Nyert: {won_str}")
        else:
            print(f"{details['season']}: Nincs egyértelmű favorit (>66%)")
    
    df_favs = pd.DataFrame(odds_summary)
    print("\n")

    # 3.5. Korreláció (Csak favorit meccseken)
    if not df_favs.empty:
        print("--- 3.5. Favorit esélye vs Rendes játékidős gólkülönbség korreláció ---")
        rt_stats = []
        for _, fav_row in df_favs.iterrows():
            match_id = fav_row['match_id']
            incidents = df_incidents[df_incidents['match_id'] == match_id]
            rt_goals = incidents[(incidents['type'] == 'goal') & (incidents['time'] <= 90)]
            
            h_g = len(rt_goals[rt_goals['is_home'] == True])
            a_g = len(rt_goals[rt_goals['is_home'] == False])
            
            rt_diff = (h_g - a_g) if fav_row['favorite'] == "Hazai" else (a_g - h_g)
            rt_stats.append({'p_fav': fav_row['p_fav'], 'rt_diff': rt_diff})
            
        df_rt = pd.DataFrame(rt_stats)
        if len(df_rt) > 1:
            print(f"Korreláció (Favorit esélye vs Rendes játékidős gólkülönbség): {df_rt['p_fav'].corr(df_rt['rt_diff']):.2f}")
        print("\n")

    # 3.6. Várható vs Tényleges (Csak favorit meccseken)
    if not df_favs.empty:
        print("--- 3.6. Várható vs Tényleges favorit kupagyőzelmek (Csak >66% esetén) ---")
        exp = df_favs['p_fav'].sum()
        act = df_favs['favorite_won'].sum()
        print(f"Várható győzelmek: {exp:.2f}")
        print(f"Tényleges győzelmek: {act}")
        print(f"Különbség: {act - exp:.2f}")
        print("\n")

    # 4. Underdog szívósság (Csak favorit meccseken)
    if not df_favs.empty:
        print("--- 4. Survival Modell: Favorit esélye vs Underdog szívósság ---")
        resilience_data = []
        for _, fav_row in df_favs.iterrows():
            match_id = fav_row['match_id']
            details = df_details[df_details['match_id'] == match_id].iloc[0]
            incidents = df_incidents[df_incidents['match_id'] == match_id].sort_values(by='total_time')
            
            fav_is_home = (fav_row['favorite'] == "Hazai")
            fav_goals = incidents[(incidents['type'] == 'goal') & 
                                 (((incidents['is_home'] == True) & fav_is_home) | 
                                  ((incidents['is_home'] == False) & (not fav_is_home)))]
            
            if fav_goals.empty:
                lasted = 120
                print(f"  {fav_row['season']} ({details['home_team'] if fav_is_home else details['away_team']} - Favorit):")
                print(f"    Az underdog nem kapott gólt. Végeredmény: {details['home_score']}-{details['away_score']}")
            else:
                lasted = fav_goals.iloc[0]['time']
                print(f"  {fav_row['season']} ({details['home_team'] if fav_is_home else details['away_team']} - Favorit):")
                print(f"    Az underdog eddig bírta: {int(lasted)} perc")
                for _, goal in fav_goals.iterrows():
                    t_str = f"{int(goal['time'])}'"
                    if pd.notna(goal['added_time']) and goal['added_time'] > 0: t_str += f"+{int(goal['added_time'])}'"
                    print(f"      Gól @ {t_str}. Állás: {int(goal['home_score'])}-{int(goal['away_score'])}")
            
            resilience_data.append({'p_fav': fav_row['p_fav'], 'lasted': lasted})
            
        df_res = pd.DataFrame(resilience_data)
        if len(df_res) > 1:
            print(f"\nKorreláció (Favorit esélye vs Underdog túlélési idő): {df_res['p_fav'].corr(df_res['lasted']):.2f}")
        print(f"Átlagos túlélési idő kiemelt favorit ellen: {df_res['lasted'].mean():.1f} perc")
        print("\n")

    # 5. A Fradi-faktor (Csak ha Fradi favorit volt >66%)
    print("--- 5. A Fradi-faktor: Hogyan nyer a dinasztia kiemelt favoritként? ---")
    fradi_matches = df_details[(df_details['home_team'] == 'Ferencváros TC') | (df_details['away_team'] == 'Ferencváros TC')]
    fradi_stats = []
    for _, row in fradi_matches.iterrows():
        match_id = row['match_id']
        fav_info = df_favs[df_favs['match_id'] == match_id]
        if fav_info.empty: continue
        
        fav_row = fav_info.iloc[0]
        fav_team = fav_row['home_team'] if fav_row['favorite'] == 'Hazai' else fav_row['away_team']
        if fav_team != 'Ferencváros TC': continue
            
        fradi_stats.append({
            'season': row['season'],
            'winner': fav_row['favorite_won'],
            'res': "Rendes játékidő" if row['status'] == "Ended" else row['status'],
            'goal_diff': abs(int(row['home_score']) - int(row['away_score']))
        })
    
    if fradi_stats:
        df_fradi = pd.DataFrame(fradi_stats)
        print(f"Fradi győzelmek favoritként (>66%): {df_fradi['winner'].sum()} / {len(df_fradi)}")
        print(f"Átlagos gólkülönbség: {df_fradi['goal_diff'].mean():.1f}")
        for _, r in df_fradi.iterrows():
            print(f"  {r['season']}: Győzelem: {'Igen' if r['winner'] else 'Nem'}, Státusz: {r['res']}, Különbség: {r['goal_diff']}")
    else:
        print("Fradi nem volt kiemelt favorit (>66%) a vizsgált döntőkben (vagy nincs adat).")
    print("\n")

    # 6. A döntők fizikája (Összes meccsen)
    print("--- 6. A döntők fizikája: Késői feszültség (Összes döntő) ---")
    total_g = len(df_incidents[df_incidents['type'] == 'goal'])
    total_c = len(df_incidents[df_incidents['type'] == 'card'])
    late = df_incidents[df_incidents['total_time'] >= 75]
    late_g = len(late[late['type'] == 'goal'])
    late_c = len(late[late['type'] == 'card'])
    
    print(f"Összes gól/lap: {total_g} / {total_c}")
    print(f"Ebből a 75. perc után: {late_g} gól ({late_g/total_g*100:.1f}%), {late_c} lap ({late_c/total_c*100:.1f}%)")
    
    # Lapok eloszlása favorit vs underdog (>66% meccseken)
    if not df_favs.empty:
        fav_c = 0
        und_c = 0
        for _, fav_row in df_favs.iterrows():
            m_id = fav_row['match_id']
            fav_is_h = (fav_row['favorite'] == "Hazai")
            m_late_c = df_incidents[(df_incidents['match_id'] == m_id) & (df_incidents['total_time'] >= 75) & (df_incidents['type'] == 'card')]
            for _, c in m_late_c.iterrows():
                if (fav_is_h and c['is_home']) or (not fav_is_h and not c['is_home']): fav_c += 1
                else: und_c += 1
        print(f"Lapok a 75' után kiemelt favorit meccseken - Favorit: {fav_c}, Underdog: {und_c}")

if __name__ == "__main__":
    run_analysis()
