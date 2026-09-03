import os
import math
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")

def run_v1_1_validation():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(by=['season_id', 'matchday', 'date', 'match_id']).reset_index(drop=True)
    
    # 1. Elo Walk-forward futtatása
    from evaluation_framework import fast_walkforward
    
    matches_data = [
        (int(r.season_id), int(r.matchday), str(r.home_team), str(r.away_team), str(r.result))
        for r in df.itertuples()
    ]
    
    # Naive baseline expanding
    h_c, d_c, a_c = 1, 1, 1
    naive_records = []
    for s_id, md, home, away, res in matches_data:
        tot = h_c + d_c + a_c
        p_h, p_d, p_a = h_c / tot, d_c / tot, a_c / tot
        p_act = p_h if res == 'H' else (p_d if res == 'D' else p_a)
        ll = -math.log(max(1e-15, p_act))
        naive_records.append({'season_id': s_id, 'p_h': p_h, 'p_d': p_d, 'p_a': p_a, 'naive_ll': ll})
        if res == 'H': h_c += 1
        elif res == 'D': d_c += 1
        else: a_c += 1
        
    df_naive = pd.DataFrame(naive_records)
    
    # Elo optimal params
    k_factor = 15.0
    home_adv = 60.0
    season_reg = 1.0
    draw_param = 0.28
    
    ratings = {}
    elo_records = []
    current_season = None
    
    for idx, (s_id, md, home, away, res) in enumerate(matches_data):
        if current_season is not None and s_id != current_season:
            for t in ratings:
                ratings[t] = 1500.0 + season_reg * (ratings[t] - 1500.0)
        current_season = s_id
        
        r_home = ratings.get(home, 1500.0)
        r_away = ratings.get(away, 1500.0)
        d_elo = (r_home + home_adv) - r_away
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        
        p_draw = draw_param * math.exp(-math.pow(d_elo / 350.0, 2))
        p_home = max(0.001, e_h - 0.5 * p_draw)
        p_away = max(0.001, (1.0 - e_h) - 0.5 * p_draw)
        p_draw = max(0.001, p_draw)
        tot = p_home + p_draw + p_away
        p_home /= tot; p_draw /= tot; p_away /= tot
        
        p_act = p_home if res == 'H' else (p_draw if res == 'D' else p_away)
        elo_ll = -math.log(max(1e-15, p_act))
        
        elo_records.append({
            'season_id': s_id,
            'matchday': md,
            'home_team': home,
            'away_team': away,
            'result': res,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
            'elo_ll': elo_ll
        })
        
        s_act = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
        delta = k_factor * (s_act - e_h)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
        
    df_elo = pd.DataFrame(elo_records)
    df_eval = pd.concat([df_elo, df_naive[['naive_ll']]], axis=1)
    
    # Különbség meccsszinten
    df_eval['diff_ll'] = df_eval['naive_ll'] - df_eval['elo_ll']
    
    # TEST SET (2022-2025)
    test_df = df_eval[df_eval['season_id'] >= 2022].copy()
    diffs = test_df['diff_ll'].values
    
    print("=================================================================")
    print("1. STATISZTIKAI SZIGNIFIKANCIA TESZT (2022-2025 TEST SET, N=792)")
    print("=================================================================")
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(len(diffs))
    
    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(test_df['naive_ll'], test_df['elo_ll'])
    
    # Wilcoxon signed-rank test
    w_stat, p_val_w = stats.wilcoxon(test_df['naive_ll'], test_df['elo_ll'])
    
    # Bootstrap 95% CI (10 000 resamples)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(10000)]
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    print(f"Átlagos meccsenkénti Log Loss előny (Mean Diff): {mean_diff:+.4f}")
    print(f"Szórás (Std): {std_diff:.4f} | Standard Error: {se_diff:.4f}")
    print(f"95% Bootstrap Konfidencia Intervallum:          [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"Paired t-test:    t = {t_stat:.3f}, p-value = {p_val_t:.4e} {'(SZIGNIFIKÁNS p < 0.01)' if p_val_t < 0.01 else ''}")
    print(f"Wilcoxon test:    W = {w_stat:.1f}, p-value = {p_val_w:.4e} {'(SZIGNIFIKÁNS p < 0.01)' if p_val_w < 0.01 else ''}")
    
    print("\n=================================================================")
    print("2. SZEZONONKÉNTI TEST SET TELJESÍTMÉNY (2022 - 2025)")
    print("=================================================================")
    s_grp = test_df.groupby('season_id').agg(
        matches=('result', 'count'),
        elo_ll=('elo_ll', 'mean'),
        naive_ll=('naive_ll', 'mean')
    ).reset_index()
    s_grp['gain_ll'] = s_grp['naive_ll'] - s_grp['elo_ll']
    s_grp['gain_pct'] = (s_grp['gain_ll'] / s_grp['naive_ll']) * 100
    print(s_grp.to_string(index=False))
    
    print("\n=================================================================")
    print("3. TELJES 3-WAY KALIBRÁCIÓ (HOME, DRAW, AWAY A 2022-2025 TESZTEN)")
    print("=================================================================")
    
    for outcome, col, target in [('HOME WIN', 'p_home', 'H'), ('DRAW', 'p_draw', 'D'), ('AWAY WIN', 'p_away', 'A')]:
        print(f"\n--- {outcome} KALIBRÁCIÓ ---")
        bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0] if outcome != 'DRAW' else [0.0, 0.20, 0.25, 0.28, 0.32, 1.0]
        test_df['bin'] = pd.cut(test_df[col], bins=bins)
        rows = []
        for interval, grp in test_df.groupby('bin', observed=False):
            if len(grp) == 0: continue
            p_mean = grp[col].mean()
            act_rate = (grp['result'] == target).mean()
            rows.append({
                'Sáv (Interval)': str(interval),
                'Meccsek': len(grp),
                'Predikált átlag': f"{p_mean*100:.1f}%",
                'Valós arány': f"{act_rate*100:.1f}%",
                'Eltérés (Rés)': f"{(act_rate - p_mean)*100:+.1f}%"
            })
        print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    run_v1_1_validation()
