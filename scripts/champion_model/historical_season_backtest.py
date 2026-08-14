import os
import math
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]

def build_dc_lookup_table(rho=-0.10, max_goals=6):
    grid = np.linspace(-1.5, 1.5, 151)
    lut_h = np.zeros((151, 151), dtype=np.float32)
    lut_d = np.zeros((151, 151), dtype=np.float32)
    
    m = max_goals + 1
    for i, log_lh in enumerate(grid):
        lh = max(0.1, min(4.5, math.exp(log_lh)))
        px = [math.exp(-lh) * (lh**x) / FACT[x] for x in range(m)]
        
        for j, log_la in enumerate(grid):
            la = max(0.1, min(4.5, math.exp(log_la)))
            py = [math.exp(-la) * (la**y) / FACT[y] for y in range(m)]
            
            tau_00 = 1.0 - lh * la * rho
            tau_01 = 1.0 + lh * rho
            tau_10 = 1.0 + la * rho
            tau_11 = 1.0 - rho
            
            ph, pd_ = 0.0, 0.0
            for x in range(m):
                px_val = px[x]
                for y in range(m):
                    prob = px_val * py[y]
                    if x == 0 and y == 0: prob *= tau_00
                    elif x == 0 and y == 1: prob *= tau_01
                    elif x == 1 and y == 0: prob *= tau_10
                    elif x == 1 and y == 1: prob *= tau_11
                    
                    if x > y: ph += prob
                    elif x == y: pd_ += prob
                    
            lut_h[i, j] = ph
            lut_d[i, j] = pd_
            
    return grid, lut_h, lut_d

GRID, LUT_H, LUT_D = build_dc_lookup_table(rho=-0.10)
GRID_MIN = -1.5
GRID_STEP = 0.02
GRID_N = 150

def get_proba_fast(log_lh, log_la):
    ih = int((log_lh - GRID_MIN) / GRID_STEP)
    ia = int((log_la - GRID_MIN) / GRID_STEP)
    ih = max(0, min(GRID_N, ih))
    ia = max(0, min(GRID_N, ia))
    ph = LUT_H[ih, ia]
    pd_ = LUT_D[ih, ia]
    return ph, pd_

def simulate_full_season_lut(fixtures_by_md, pre_alpha, pre_beta, teams, 
                             lr_normal=0.02, lr_early_mult=2.5, early_md_cutoff=5,
                             home_adv=0.25, base_mu=0.30,
                             is_dynamic_in_sim=True, n_sims=3000):
    sim_final_pts = {t: np.zeros(n_sims, dtype=np.int16) for t in teams}
    sim_final_ranks = {t: np.zeros(n_sims, dtype=np.int8) for t in teams}
    
    total_matches = sum(len(f) for f in fixtures_by_md.values())
    rnds = np.random.rand(n_sims, total_matches)
    
    for s_i in range(n_sims):
        alpha = dict(pre_alpha)
        beta  = dict(pre_beta)
        
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        match_counter = 0
        
        for md, md_fixtures in fixtures_by_md.items():
            lr = (lr_normal * lr_early_mult) if (md <= early_md_cutoff) else lr_normal
            
            for h, a in md_fixtures:
                log_lh = base_mu + alpha.get(h, 0.0) + beta.get(a, 0.0) + home_adv
                log_la = base_mu + alpha.get(a, 0.0) + beta.get(h, 0.0)
                
                ph, pd_ = get_proba_fast(log_lh, log_la)
                rnd = rnds[s_i, match_counter]
                match_counter += 1
                
                if rnd < ph:
                    pts[h] += 3
                    gd[h] += 1
                    gd[a] -= 1
                    err_h, err_a = 0.8, -0.6
                elif rnd < ph + pd_:
                    pts[h] += 1
                    pts[a] += 1
                    err_h, err_a = -0.2, -0.2
                else:
                    pts[a] += 3
                    gd[a] += 1
                    gd[h] -= 1
                    err_h, err_a = -0.6, 0.8
                    
                if is_dynamic_in_sim:
                    alpha[h] = alpha.get(h, 0.0) + lr * err_h
                    beta[a]  = beta.get(a, 0.0)  + lr * err_h
                    alpha[a] = alpha.get(a, 0.0) + lr * err_a
                    beta[h]  = beta.get(h, 0.0)  + lr * err_a
                    
        sorted_t = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        for rank, t in enumerate(sorted_t, 1):
            sim_final_pts[t][s_i] = pts[t]
            sim_final_ranks[t][s_i] = rank
            
    return sim_final_pts, sim_final_ranks

def run_historical_season_benchmark():
    df = pd.read_csv(CSV_PATH)
    df['parsed_date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['season_id', 'matchday', 'parsed_date', 'match_id']).reset_index(drop=True)
    
    matches = []
    for r in df.itertuples():
        matches.append({
            'season_id': int(r.season_id),
            'matchday': int(r.matchday),
            'date': r.parsed_date,
            'home_team': str(r.home_team),
            'away_team': str(r.away_team),
            'result': str(r.result),
            'home_score': int(r.home_score),
            'away_score': int(r.away_score)
        })
        
    seasons_to_eval = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    print("=================================================================")
    print("HISTORICAL SEASON-LEVEL MONTE CARLO BENCHMARK (2018 - 2025)")
    print("=================================================================")
    print(f"Evaluating {len(seasons_to_eval)} full seasons with 3,000 sequential simulations each...\n")
    
    model_configs = [
        {
            'name': 'Model 1: Static Poisson (Baseline)',
            'is_dynamic': False,
            'use_early_boost': False
        },
        {
            'name': 'Model 2: Dynamic DC (Static In-Sim)',
            'is_dynamic': False,
            'use_early_boost': False
        },
        {
            'name': 'Model 3: Dynamic DC (Sequential In-Sim Updates)',
            'is_dynamic': True,
            'use_early_boost': False
        },
        {
            'name': 'Model 4: Adaptive Dynamic DC (Early-Season Boost + In-Sim)',
            'is_dynamic': True,
            'use_early_boost': True
        }
    ]
    
    benchmark_results = []
    np.random.seed(42)
    
    for cfg in model_configs:
        m_name = cfg['name']
        print(f"--> Running {m_name} across 8 historical seasons...")
        
        all_champ_brier = []
        all_top3_brier = []
        all_releg_brier = []
        all_points_mae = []
        all_rank_corrs = []
        
        for target_s in seasons_to_eval:
            alpha, beta = {}, {}
            for m in matches:
                if m['season_id'] < target_s:
                    lh = max(0.1, min(4.5, math.exp(0.30 + alpha.get(m['home_team'], 0.0) + beta.get(m['away_team'], 0.0) + 0.25)))
                    la = max(0.1, min(4.5, math.exp(0.30 + alpha.get(m['away_team'], 0.0) + beta.get(m['home_team'], 0.0))))
                    alpha[m['home_team']] = alpha.get(m['home_team'], 0.0) + 0.02 * (m['home_score'] - lh)
                    beta[m['away_team']]  = beta.get(m['away_team'], 0.0)  + 0.02 * (m['home_score'] - lh)
                    alpha[m['away_team']] = alpha.get(m['away_team'], 0.0) + 0.02 * (m['away_score'] - la)
                    beta[m['home_team']]  = beta.get(m['home_team'], 0.0)  + 0.02 * (m['away_score'] - la)
                    
            s_matches = [m for m in matches if m['season_id'] == target_s]
            teams = sorted(list(set([m['home_team'] for m in s_matches])))
            
            fixtures_by_md = {}
            for m in s_matches:
                fixtures_by_md.setdefault(m['matchday'], []).append((m['home_team'], m['away_team']))
                
            act_pts = {t: 0 for t in teams}
            act_gd  = {t: 0 for t in teams}
            for m in s_matches:
                h, a, res, hs, ascore = m['home_team'], m['away_team'], m['result'], m['home_score'], m['away_score']
                if res == 'H': act_pts[h] += 3
                elif res == 'D': act_pts[h] += 1; act_pts[a] += 1
                else: act_pts[a] += 3
                act_gd[h] += (hs - ascore)
                act_gd[a] += (ascore - hs)
                
            actual_ranks = {t: r for r, t in enumerate(sorted(teams, key=lambda t: (act_pts[t], act_gd[t]), reverse=True), 1)}
            
            sim_pts, sim_ranks = simulate_full_season_lut(
                fixtures_by_md, alpha, beta, teams,
                lr_normal=0.02,
                lr_early_mult=2.5 if cfg['use_early_boost'] else 1.0,
                early_md_cutoff=5,
                home_adv=0.25,
                base_mu=0.30,
                is_dynamic_in_sim=cfg['is_dynamic'],
                n_sims=3000
            )
            
            exp_pts = {t: np.mean(sim_pts[t]) for t in teams}
            exp_ranks = {t: np.mean(sim_ranks[t]) for t in teams}
            p_champ = {t: np.mean(sim_ranks[t] == 1) for t in teams}
            p_top3  = {t: np.mean(sim_ranks[t] <= 3) for t in teams}
            p_releg = {t: np.mean(sim_ranks[t] >= 11) for t in teams}
            
            mae_s = np.mean([abs(exp_pts[t] - act_pts[t]) for t in teams])
            all_points_mae.append(mae_s)
            
            r_corr, _ = stats.spearmanr([exp_ranks[t] for t in teams], [actual_ranks[t] for t in teams])
            if not np.isnan(r_corr):
                all_rank_corrs.append(r_corr)
            
            cb_s = np.mean([(p_champ[t] - (1.0 if actual_ranks[t] == 1 else 0.0))**2 for t in teams])
            all_champ_brier.append(cb_s)
            
            t3_s = np.mean([(p_top3[t] - (1.0 if actual_ranks[t] <= 3 else 0.0))**2 for t in teams])
            all_top3_brier.append(t3_s)
            
            rel_s = np.mean([(p_releg[t] - (1.0 if actual_ranks[t] >= 11 else 0.0))**2 for t in teams])
            all_releg_brier.append(rel_s)
            
        benchmark_results.append({
            'Model Name': m_name,
            'Points MAE': round(float(np.mean(all_points_mae)), 2),
            'Rank Corr': round(float(np.mean(all_rank_corrs)), 3),
            'Champion Brier': round(float(np.mean(all_champ_brier)), 4),
            'Top 3 Brier': round(float(np.mean(all_top3_brier)), 4),
            'Relegation Brier': round(float(np.mean(all_releg_brier)), 4)
        })
        
    df_bench = pd.DataFrame(benchmark_results)
    print("\n" + "="*95)
    print("HISTORICAL SEASON BENCHMARK LEADERBOARD (AVERAGE ACROSS 2018-2025, 8 SEASONS)")
    print("="*95)
    print(df_bench.to_string(index=False))

if __name__ == "__main__":
    run_historical_season_benchmark()
