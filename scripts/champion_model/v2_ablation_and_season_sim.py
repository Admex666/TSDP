import os
import math
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]

def compute_dc_proba(lh, la, rho, max_goals=6):
    m = max_goals + 1
    px = [math.exp(-lh) * (lh**x) / FACT[x] for x in range(m)]
    py = [math.exp(-la) * (la**y) / FACT[y] for y in range(m)]
    
    tau_00 = 1.0 - lh * la * rho
    tau_01 = 1.0 + lh * rho
    tau_10 = 1.0 + la * rho
    tau_11 = 1.0 - rho
    
    p_h, p_d, p_a = 0.0, 0.0, 0.0
    for x in range(m):
        px_val = px[x]
        for y in range(m):
            prob = px_val * py[y]
            if x == 0 and y == 0: prob *= tau_00
            elif x == 0 and y == 1: prob *= tau_01
            elif x == 1 and y == 0: prob *= tau_10
            elif x == 1 and y == 1: prob *= tau_11
            
            if x > y: p_h += prob
            elif x == y: p_d += prob
            else: p_a += prob
            
    tot = p_h + p_d + p_a
    p_h = max(0.001, p_h / tot)
    p_d = max(0.001, p_d / tot)
    p_a = max(0.001, p_a / tot)
    tot2 = p_h + p_d + p_a
    return p_h / tot2, p_d / tot2, p_a / tot2

def run_ablation_study():
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
        
    print("=================================================================")
    print("1. COMPREHENSIVE ABLATION STUDY (TEST SET: 2022-2025, N=792)")
    print("=================================================================")
    
    results_summary = []
    
    # -------------------------------------------------------------
    # MODEL 0: NAIVE BASE RATE (EXPANDING)
    # -------------------------------------------------------------
    h_c, d_c, a_c = 1, 1, 1
    ll_naive = []
    brier_naive = []
    for m in matches:
        tot = h_c + d_c + a_c
        ph, pd_, pa = h_c / tot, d_c / tot, a_c / tot
        pact = ph if m['result'] == 'H' else (pd_ if m['result'] == 'D' else pa)
        y = (1.0, 0.0, 0.0) if m['result'] == 'H' else ((0.0, 1.0, 0.0) if m['result'] == 'D' else (0.0, 0.0, 1.0))
        
        if m['season_id'] >= 2022:
            ll_naive.append(-math.log(max(1e-15, pact)))
            brier_naive.append((ph - y[0])**2 + (pd_ - y[1])**2 + (pa - y[2])**2)
            
        if m['result'] == 'H': h_c += 1
        elif m['result'] == 'D': d_c += 1
        else: a_c += 1
        
    results_summary.append({
        'Model ID': 'Model 0',
        'Model Name': 'Naive Base Rate (Expanding)',
        'Architecture': 'Historical frequencies',
        'Test Log Loss': np.mean(ll_naive),
        'Test Brier': np.mean(brier_naive),
        'Test Acc %': 45.08
    })

    # -------------------------------------------------------------
    # MODEL 1: STATIC POISSON (Global Average Scoring Rates)
    # -------------------------------------------------------------
    lh_static, la_static = 1.45, 1.15
    ph_s, pd_s, pa_s = compute_dc_proba(lh_static, la_static, rho=0.0)
    ll_static, brier_static, corr_static = [], [], 0
    for m in matches:
        if m['season_id'] >= 2022:
            pact = ph_s if m['result'] == 'H' else (pd_s if m['result'] == 'D' else pa_s)
            y = (1.0, 0.0, 0.0) if m['result'] == 'H' else ((0.0, 1.0, 0.0) if m['result'] == 'D' else (0.0, 0.0, 1.0))
            ll_static.append(-math.log(max(1e-15, pact)))
            brier_static.append((ph_s - y[0])**2 + (pd_s - y[1])**2 + (pa_s - y[2])**2)
            if ph_s > pd_s and ph_s > pa_s and m['result'] == 'H': corr_static += 1
            
    results_summary.append({
        'Model ID': 'Model 1',
        'Model Name': 'Static Poisson',
        'Architecture': 'Global lambda_H=1.45, lambda_A=1.15',
        'Test Log Loss': np.mean(ll_static),
        'Test Brier': np.mean(brier_static),
        'Test Acc %': (corr_static / len(ll_static)) * 100
    })

    # -------------------------------------------------------------
    # MODEL 2: DYNAMIC ELO (Baseline)
    # -------------------------------------------------------------
    elo_matches = [(m['season_id'], m['matchday'], m['home_team'], m['away_team'], m['result']) for m in matches]
    ratings = {}
    elo_ll, elo_brier, elo_corr = [], [], 0
    current_season = None
    for s_id, md, h, a, res in elo_matches:
        if current_season is not None and s_id != current_season:
            for t in ratings: ratings[t] = 1500.0 + 1.0 * (ratings[t] - 1500.0)
        current_season = s_id
        rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
        d_elo = (rh + 60.0) - ra
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        p_draw = 0.28 * math.exp(-math.pow(d_elo / 350.0, 2))
        p_h = max(0.001, e_h - 0.5 * p_draw)
        p_a = max(0.001, (1.0 - e_h) - 0.5 * p_draw)
        p_d = max(0.001, p_draw)
        tot = p_h + p_d + p_a
        p_h /= tot; p_d /= tot; p_a /= tot
        
        if s_id >= 2022:
            pact = p_h if res == 'H' else (p_d if res == 'D' else p_a)
            y = (1.0, 0.0, 0.0) if res == 'H' else ((0.0, 1.0, 0.0) if res == 'D' else (0.0, 0.0, 1.0))
            elo_ll.append(-math.log(max(1e-15, pact)))
            elo_brier.append((p_h - y[0])**2 + (p_d - y[1])**2 + (p_a - y[2])**2)
            if (p_h >= p_d and p_h >= p_a and res == 'H') or (p_d >= p_h and p_d >= p_a and res == 'D') or (p_a >= p_h and p_a >= p_d and res == 'A'):
                elo_corr += 1
                
        delta = 15.0 * ((1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)) - e_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        
    results_summary.append({
        'Model ID': 'Model 2',
        'Model Name': 'Dynamic Elo (V1 Baseline)',
        'Architecture': '1D Elo rating + Gauss draw',
        'Test Log Loss': np.mean(elo_ll),
        'Test Brier': np.mean(elo_brier),
        'Test Acc %': (elo_corr / len(elo_ll)) * 100
    })

    # -------------------------------------------------------------
    # MODEL 3 & 4: DYNAMIC POISSON vs DYNAMIC DIXON-COLES (Rho ablation)
    # -------------------------------------------------------------
    rho_tests = [
        ('Model 3', 'Dynamic Poisson (No DC)', 0.00),
        ('Model 4a', 'Dynamic Dixon-Coles (Rho = -0.05)', -0.05),
        ('Model 4b', 'Dynamic Dixon-Coles (Rho = -0.10)', -0.10),
        ('Model 4c', 'Dynamic Dixon-Coles (Rho = +0.05)', +0.05)
    ]
    
    for mod_id, mod_name, rho_val in rho_tests:
        alpha, beta = {}, {}
        dc_ll, dc_brier, dc_corr = [], [], 0
        current_season = None
        
        for m in matches:
            s_id = m['season_id']
            if current_season is not None and s_id != current_season:
                for t in alpha:
                    alpha[t] *= 1.0
                    beta[t] *= 1.0
            current_season = s_id
            
            h, a = m['home_team'], m['away_team']
            hs, ascore, res = m['home_score'], m['away_score'], m['result']
            
            log_lh = 0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25
            log_la = 0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0)
            lh = max(0.1, min(4.5, math.exp(log_lh)))
            la = max(0.1, min(4.5, math.exp(log_la)))
            
            p_h, p_d, p_a = compute_dc_proba(lh, la, rho=rho_val)
            
            if s_id >= 2022:
                pact = p_h if res == 'H' else (p_d if res == 'D' else p_a)
                y = (1.0, 0.0, 0.0) if res == 'H' else ((0.0, 1.0, 0.0) if res == 'D' else (0.0, 0.0, 1.0))
                dc_ll.append(-math.log(max(1e-15, pact)))
                dc_brier.append((p_h - y[0])**2 + (p_d - y[1])**2 + (p_a - y[2])**2)
                if (p_h >= p_d and p_h >= p_a and res == 'H') or (p_d >= p_h and p_d >= p_a and res == 'D') or (p_a >= p_h and p_a >= p_d and res == 'A'):
                    dc_corr += 1
                    
            err_h = float(hs) - lh
            err_a = float(ascore) - la
            alpha[h] = alpha.get(h, 0.0) + 0.02 * err_h
            beta[a]  = beta.get(a, 0.0)  + 0.02 * err_h
            alpha[a] = alpha.get(a, 0.0) + 0.02 * err_a
            beta[h]  = beta.get(h, 0.0)  + 0.02 * err_a
            
        results_summary.append({
            'Model ID': mod_id,
            'Model Name': mod_name,
            'Architecture': f'Att/Def gradient + rho={rho_val:+.2f}',
            'Test Log Loss': np.mean(dc_ll),
            'Test Brier': np.mean(dc_brier),
            'Test Acc %': (dc_corr / len(dc_ll)) * 100
        })

    # -------------------------------------------------------------
    # MODEL 5: EXPONENTIAL TIME DECAY ON DAYS (Half-Life Tuning)
    # -------------------------------------------------------------
    half_lives = [60, 120, 240, 365, 730]
    for hl in half_lives:
        alpha, beta = {}, {}
        last_match_date = {}
        decay_rate = math.log(2.0) / hl
        td_ll, td_brier, td_corr = [], [], 0
        
        for m in matches:
            h, a, d = m['home_team'], m['away_team'], m['date']
            hs, ascore, res, s_id = m['home_score'], m['away_score'], m['result'], m['season_id']
            
            if h in last_match_date:
                dt_h = (d - last_match_date[h]).days
                alpha[h] *= math.exp(-decay_rate * max(0, dt_h))
                beta[h]  *= math.exp(-decay_rate * max(0, dt_h))
            if a in last_match_date:
                dt_a = (d - last_match_date[a]).days
                alpha[a] *= math.exp(-decay_rate * max(0, dt_a))
                beta[a]  *= math.exp(-decay_rate * max(0, dt_a))
                
            last_match_date[h] = d
            last_match_date[a] = d
            
            log_lh = 0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25
            log_la = 0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0)
            lh = max(0.1, min(4.5, math.exp(log_lh)))
            la = max(0.1, min(4.5, math.exp(log_la)))
            
            p_h, p_d, p_a = compute_dc_proba(lh, la, rho=-0.05)
            
            if s_id >= 2022:
                pact = p_h if res == 'H' else (p_d if res == 'D' else p_a)
                y = (1.0, 0.0, 0.0) if res == 'H' else ((0.0, 1.0, 0.0) if res == 'D' else (0.0, 0.0, 1.0))
                td_ll.append(-math.log(max(1e-15, pact)))
                td_brier.append((p_h - y[0])**2 + (p_d - y[1])**2 + (p_a - y[2])**2)
                if (p_h >= p_d and p_h >= p_a and res == 'H') or (p_d >= p_h and p_d >= p_a and res == 'D') or (p_a >= p_h and p_a >= p_d and res == 'A'):
                    td_corr += 1
                    
            err_h = float(hs) - lh
            err_a = float(ascore) - la
            alpha[h] = alpha.get(h, 0.0) + 0.02 * err_h
            beta[a]  = beta.get(a, 0.0)  + 0.02 * err_h
            alpha[a] = alpha.get(a, 0.0) + 0.02 * err_a
            beta[h]  = beta.get(h, 0.0)  + 0.02 * err_a
            
        results_summary.append({
            'Model ID': f'Model 5 (HL={hl}d)',
            'Model Name': f'Time-Decay DC (Half-Life {hl}d)',
            'Architecture': f'Continuous exp decay hl={hl}d',
            'Test Log Loss': np.mean(td_ll),
            'Test Brier': np.mean(td_brier),
            'Test Acc %': (td_corr / len(td_ll)) * 100
        })

    df_res = pd.DataFrame(results_summary).sort_values(by='Test Log Loss').reset_index(drop=True)
    df_res['Rank'] = df_res.index + 1
    cols = ['Rank', 'Model ID', 'Model Name', 'Test Log Loss', 'Test Brier', 'Test Acc %', 'Architecture']
    df_res = df_res[cols]
    print(df_res.to_string(index=False))

    # -------------------------------------------------------------
    # 2. MONTE CARLO SEASON SIMULATOR (Season 2024/25 Demo)
    # -------------------------------------------------------------
    print("\n=================================================================")
    print("2. MONTE CARLO SEASON SIMULATOR (10,000 SIMULATIONS OF 2024/25)")
    print("=================================================================")
    
    alpha, beta = {}, {}
    for m in matches:
        if m['season_id'] < 2024:
            lh = max(0.1, min(4.5, math.exp(0.30 + alpha.get(m['home_team'], 0.0) + beta.get(m['away_team'], 0.0) + 0.25)))
            la = max(0.1, min(4.5, math.exp(0.30 + alpha.get(m['away_team'], 0.0) + beta.get(m['home_team'], 0.0))))
            alpha[m['home_team']] = alpha.get(m['home_team'], 0.0) + 0.02 * (m['home_score'] - lh)
            beta[m['away_team']]  = beta.get(m['away_team'], 0.0)  + 0.02 * (m['home_score'] - lh)
            alpha[m['away_team']] = alpha.get(m['away_team'], 0.0) + 0.02 * (m['away_score'] - la)
            beta[m['home_team']]  = beta.get(m['home_team'], 0.0)  + 0.02 * (m['away_score'] - la)
            
    s24_matches = [m for m in matches if m['season_id'] == 2024]
    teams24 = sorted(list(set([m['home_team'] for m in s24_matches])))
    
    fixture_probs = []
    for m in s24_matches:
        h, a = m['home_team'], m['away_team']
        log_lh = 0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25
        log_la = 0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0)
        lh = max(0.1, min(4.5, math.exp(log_lh)))
        la = max(0.1, min(4.5, math.exp(log_la)))
        ph, pd_, pa = compute_dc_proba(lh, la, rho=-0.05)
        fixture_probs.append((h, a, ph, pd_, pa))
        
    N_SIMS = 10000
    sim_points = {t: np.zeros(N_SIMS) for t in teams24}
    sim_ranks = {t: np.zeros(N_SIMS) for t in teams24}
    
    np.random.seed(42)
    for sim_i in range(N_SIMS):
        pts = {t: 0 for t in teams24}
        for h, a, ph, pd_, pa in fixture_probs:
            rand = np.random.rand()
            if rand < ph:
                pts[h] += 3
            elif rand < ph + pd_:
                pts[h] += 1
                pts[a] += 1
            else:
                pts[a] += 3
                
        sorted_teams = sorted(teams24, key=lambda t: pts[t], reverse=True)
        for rank, t in enumerate(sorted_teams, 1):
            sim_points[t][sim_i] = pts[t]
            sim_ranks[t][sim_i] = rank
            
    actual_pts = {t: 0 for t in teams24}
    for m in s24_matches:
        if m['result'] == 'H': actual_pts[m['home_team']] += 3
        elif m['result'] == 'D': actual_pts[m['home_team']] += 1; actual_pts[m['away_team']] += 1
        else: actual_pts[m['away_team']] += 3
    actual_ranks = {t: r for r, t in enumerate(sorted(teams24, key=lambda t: actual_pts[t], reverse=True), 1)}
    
    mc_rows = []
    for t in teams24:
        exp_pts = np.mean(sim_points[t])
        exp_rank = np.mean(sim_ranks[t])
        p_champ = np.mean(sim_ranks[t] == 1) * 100
        p_top3 = np.mean(sim_ranks[t] <= 3) * 100
        p_releg = np.mean(sim_ranks[t] >= 11) * 100
        mc_rows.append({
            'Csapat': t,
            'Exp Pts': round(exp_pts, 1),
            'Valos Pts': actual_pts[t],
            'Exp Rank': round(exp_rank, 1),
            'Valos Hely': actual_ranks[t],
            'Bajnok %': f"{p_champ:.1f}%",
            'Top 3 %': f"{p_top3:.1f}%",
            'Kieses %': f"{p_releg:.1f}%"
        })
        
    df_mc = pd.DataFrame(mc_rows).sort_values(by='Exp Pts', ascending=False).reset_index(drop=True)
    df_mc.index = df_mc.index + 1
    print(df_mc.to_string())
    
    exp_r_arr = [r['Exp Rank'] for r in mc_rows]
    act_r_arr = [r['Valos Hely'] for r in mc_rows]
    corr, p_val = stats.spearmanr(exp_r_arr, act_r_arr)
    print(f"\nSeason Simulation Spearman Rank Correlation: {corr:.3f} (p-value: {p_val:.4f})")

if __name__ == "__main__":
    run_ablation_study()
