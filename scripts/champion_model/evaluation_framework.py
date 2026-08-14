import os
import math
import itertools
import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")

def fast_walkforward(matches_data, k_factor=20.0, home_adv=60.0, season_reg=0.85, draw_param=0.26, max_season=2025):
    """
    Ultra-fast walk-forward loop using pure Python dicts and math functions.
    matches_data: list of tuples (season_id, matchday, home_team, away_team, result)
    """
    ratings = {}
    total_log_loss = 0.0
    total_brier = 0.0
    correct_count = 0
    count = 0
    current_season = None
    
    for season_id, matchday, home_team, away_team, result in matches_data:
        if season_id > max_season:
            break
            
        # Season regression
        if current_season is not None and season_id != current_season:
            for t in ratings:
                ratings[t] = 1500.0 + season_reg * (ratings[t] - 1500.0)
        current_season = season_id
        
        r_home = ratings.get(home_team, 1500.0)
        r_away = ratings.get(away_team, 1500.0)
        
        # Effective ratings & difference
        d_elo = (r_home + home_adv) - r_away
        
        # Expected Home Score E_H
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        
        # Draw probability
        p_draw = draw_param * math.exp(-math.pow(d_elo / 350.0, 2))
        p_home = max(0.001, e_h - 0.5 * p_draw)
        p_away = max(0.001, (1.0 - e_h) - 0.5 * p_draw)
        p_draw = max(0.001, p_draw)
        
        tot = p_home + p_draw + p_away
        p_home /= tot
        p_draw /= tot
        p_away /= tot
        
        if result == 'H':
            p_act = p_home
            s_act = 1.0
            y = (1.0, 0.0, 0.0)
        elif result == 'D':
            p_act = p_draw
            s_act = 0.5
            y = (0.0, 1.0, 0.0)
        else:
            p_act = p_away
            s_act = 0.0
            y = (0.0, 0.0, 1.0)
            
        total_log_loss += -math.log(max(1e-15, p_act))
        total_brier += (p_home - y[0])**2 + (p_draw - y[1])**2 + (p_away - y[2])**2
        
        if (p_home >= p_draw and p_home >= p_away and result == 'H') or \
           (p_draw >= p_home and p_draw >= p_away and result == 'D') or \
           (p_away >= p_home and p_away >= p_draw and result == 'A'):
            correct_count += 1
            
        count += 1
        
        # Update ratings
        delta = k_factor * (s_act - e_h)
        ratings[home_team] = r_home + delta
        ratings[away_team] = r_away - delta
        
    return {
        'log_loss': total_log_loss / count,
        'brier_score': total_brier / count,
        'accuracy': correct_count / count,
        'count': count
    }

def run_full_evaluation():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(by=['season_id', 'matchday', 'date', 'match_id']).reset_index(drop=True)
    
    matches_data = [
        (int(r.season_id), int(r.matchday), str(r.home_team), str(r.away_team), str(r.result))
        for r in df.itertuples()
    ]
    train_data = [m for m in matches_data if m[0] <= 2021]
    
    print("=================================================================")
    print("1. NAIVE BASELINE (Prior Historical Class Distribution)")
    print("=================================================================")
    h_c, d_c, a_c = 1, 1, 1
    ll_train_naive, ll_test_naive, ll_all_naive = [], [], []
    
    for season_id, matchday, home, away, result in matches_data:
        tot = h_c + d_c + a_c
        p_h, p_d, p_a = h_c / tot, d_c / tot, a_c / tot
        p_act = p_h if result == 'H' else (p_d if result == 'D' else p_a)
        loss = -math.log(max(1e-15, p_act))
        
        ll_all_naive.append(loss)
        if season_id <= 2021:
            ll_train_naive.append(loss)
        else:
            ll_test_naive.append(loss)
            
        if result == 'H': h_c += 1
        elif result == 'D': d_c += 1
        else: a_c += 1
        
    print(f"Overall Naive Log Loss (2015-2025): {np.mean(ll_all_naive):.4f}")
    print(f"Train Naive Log Loss   (2015-2021): {np.mean(ll_train_naive):.4f}")
    print(f"Test Naive Log Loss    (2022-2025): {np.mean(ll_test_naive):.4f}")
    
    # 2. TUNING ON TRAIN SET ONLY (2015-2021)
    print("\n=================================================================")
    print("2. HYPERPARAMETER TUNING STRICTLY ON 2015-2021 (7 SEASONS, 1386 MATCHES)")
    print("=================================================================")
    
    k_grid = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    h_grid = [20.0, 40.0, 60.0, 80.0, 100.0]
    rho_grid = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    draw_grid = [0.22, 0.24, 0.26, 0.28, 0.30]
    
    search_space = list(itertools.product(k_grid, h_grid, rho_grid, draw_grid))
    best_train_ll = float("inf")
    best_params = None
    
    for k, h, rho, d in search_space:
        res = fast_walkforward(train_data, k_factor=k, home_adv=h, season_reg=rho, draw_param=d)
        if res['log_loss'] < best_train_ll:
            best_train_ll = res['log_loss']
            best_params = {"k_factor": k, "home_advantage": h, "season_regression": rho, "draw_param": d}
            
    print(f"Optimal Parameters Found on Training Period (2015-2021):")
    print(f" - K-Factor:         {best_params['k_factor']}")
    print(f" - Home Advantage:   {best_params['home_advantage']} Elo points")
    print(f" - Season Reg (Rho): {best_params['season_regression']*100:.0f}% (Rho={best_params['season_regression']})")
    print(f" - Draw Param:       {best_params['draw_param']}")
    print(f" - Train Log Loss:   {best_train_ll:.4f} (vs Naive: {np.mean(ll_train_naive):.4f})")
    
    # 3. OUT-OF-SAMPLE TEST EVALUATION (2022-2025)
    print("\n=================================================================")
    print("3. OUT-OF-SAMPLE TEST EVALUATION (2022-2025, 4 SEASONS, 792 MATCHES)")
    print("=================================================================")
    
    # Run continuous walkforward across all seasons with optimal params
    # and record predictions specifically on test set
    ratings = {}
    test_records = []
    current_season = None
    
    for season_id, matchday, home_team, away_team, result in matches_data:
        if current_season is not None and season_id != current_season:
            for t in ratings:
                ratings[t] = 1500.0 + best_params['season_regression'] * (ratings[t] - 1500.0)
        current_season = season_id
        
        r_home = ratings.get(home_team, 1500.0)
        r_away = ratings.get(away_team, 1500.0)
        d_elo = (r_home + best_params['home_advantage']) - r_away
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        
        p_draw = best_params['draw_param'] * math.exp(-math.pow(d_elo / 350.0, 2))
        p_home = max(0.001, e_h - 0.5 * p_draw)
        p_away = max(0.001, (1.0 - e_h) - 0.5 * p_draw)
        p_draw = max(0.001, p_draw)
        tot = p_home + p_draw + p_away
        p_home /= tot; p_draw /= tot; p_away /= tot
        
        if result == 'H':
            p_act = p_home
            s_act = 1.0
            y = (1.0, 0.0, 0.0)
        elif result == 'D':
            p_act = p_draw
            s_act = 0.5
            y = (0.0, 1.0, 0.0)
        else:
            p_act = p_away
            s_act = 0.0
            y = (0.0, 0.0, 1.0)
            
        log_loss_i = -math.log(max(1e-15, p_act))
        brier_i = (p_home - y[0])**2 + (p_draw - y[1])**2 + (p_away - y[2])**2
        is_corr = 1 if (p_home >= p_draw and p_home >= p_away and result == 'H') or \
                       (p_draw >= p_home and p_draw >= p_away and result == 'D') or \
                       (p_away >= p_home and p_away >= p_draw and result == 'A') else 0
                       
        if season_id >= 2022:
            test_records.append({
                'season_id': season_id,
                'matchday': matchday,
                'home_team': home_team,
                'away_team': away_team,
                'result': result,
                'r_home': r_home,
                'r_away': r_away,
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'log_loss': log_loss_i,
                'brier_score': brier_i,
                'is_correct': is_corr
            })
            
        delta = best_params['k_factor'] * (s_act - e_h)
        ratings[home_team] = r_home + delta
        ratings[away_team] = r_away - delta
        
    df_test = pd.DataFrame(test_records)
    test_elo_ll = df_test['log_loss'].mean()
    test_naive_ll = np.mean(ll_test_naive)
    test_brier = df_test['brier_score'].mean()
    test_acc = df_test['is_correct'].mean()
    
    print(f"Test Set Performance (UNTOUCHED 2022-2025):")
    print(f" - Elo Test Log Loss:     {test_elo_ll:.4f}")
    print(f" - Naive Test Log Loss:   {test_naive_ll:.4f}")
    print(f" - Log Loss Edge / Gain:  {test_naive_ll - test_elo_ll:+.4f} ({(test_naive_ll - test_elo_ll)/test_naive_ll*100:+.2f}%)")
    print(f" - Test Brier Score:      {test_brier:.4f}")
    print(f" - Test Accuracy:         {test_acc*100:.2f}%")
    
    # 4. CALIBRATION ANALYSIS (Reliability on Test Set)
    print("\n=================================================================")
    print("4. CALIBRATION ANALYSIS (Reliability on 2022-2025 Test Set)")
    print("=================================================================")
    
    bins = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0]
    df_test['p_home_bin'] = pd.cut(df_test['p_home'], bins=bins)
    
    calib_rows = []
    for interval, grp in df_test.groupby('p_home_bin', observed=False):
        n_matches = len(grp)
        if n_matches == 0: continue
        pred_mean = grp['p_home'].mean()
        actual_win_rate = (grp['result'] == 'H').mean()
        calib_rows.append({
            'P(Home) Bin': str(interval),
            'Matches': n_matches,
            'Pred P(Home)': f"{pred_mean*100:.1f}%",
            'Actual Win %': f"{actual_win_rate*100:.1f}%",
            'Calibration Gap': f"{(actual_win_rate - pred_mean)*100:+.1f}%"
        })
        
    print(pd.DataFrame(calib_rows).to_string(index=False))

if __name__ == "__main__":
    run_full_evaluation()
