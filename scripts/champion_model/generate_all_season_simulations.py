import os
import math
import numpy as np
import pandas as pd
from datetime import datetime

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
    return LUT_H[ih, ia], LUT_D[ih, ia]

def generate_all_season_data(n_sims=2000):
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
        
    all_seasons = sorted(list(set(m['season_id'] for m in matches)))
    print(f"Generating simulations across {len(all_seasons)} seasons: {all_seasons}...")
    
    records = []
    
    # Sequential warm-up of dynamic ratings
    alpha = {}
    beta = {}
    
    for s_idx, target_s in enumerate(all_seasons):
        print(f"--> Simulating Season {target_s}/{str(target_s+1)[-2:]}...")
        
        # Szezon kezdete előtti rating másolat
        pre_season_alpha = dict(alpha)
        pre_season_beta  = dict(beta)
        
        s_matches = [m for m in matches if m['season_id'] == target_s]
        teams = sorted(list(set([m['home_team'] for m in s_matches])))
        max_md = max(m['matchday'] for m in s_matches)
        
        # Track standings
        actual_pts = {t: 0 for t in teams}
        actual_gd  = {t: 0 for t in teams}
        actual_gf  = {t: 0 for t in teams}
        actual_played = {t: 0 for t in teams}
        
        curr_alpha = dict(pre_season_alpha)
        curr_beta  = dict(pre_season_beta)
        
        # Matchday dates map
        md_dates = {}
        for m in s_matches:
            d_str = m['date'].strftime('%Y-%m-%d')
            md_dates.setdefault(m['matchday'], []).append(d_str)
        # Matchday canonical date (last date of matchday)
        md_last_date = {md: max(d_list) for md, d_list in md_dates.items()}
        # Pre-season date (day before first match)
        first_date = min(m['date'] for m in s_matches)
        md_last_date[0] = (first_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        for md in range(0, max_md + 1):
            if md > 0:
                md_matches = [m for m in s_matches if m['matchday'] == md]
                for m in md_matches:
                    h, a, res, hs, ascore = m['home_team'], m['away_team'], m['result'], m['home_score'], m['away_score']
                    actual_played[h] += 1
                    actual_played[a] += 1
                    if res == 'H':
                        actual_pts[h] += 3
                    elif res == 'D':
                        actual_pts[h] += 1
                        actual_pts[a] += 1
                    else:
                        actual_pts[a] += 3
                    actual_gd[h] += (hs - ascore)
                    actual_gd[a] += (ascore - hs)
                    actual_gf[h] += hs
                    actual_gf[a] += ascore
                    
                    lh = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(h, 0.0) + curr_beta.get(a, 0.0) + 0.25)))
                    la = max(0.1, min(4.5, math.exp(0.30 + curr_alpha.get(a, 0.0) + curr_beta.get(h, 0.0))))
                    curr_alpha[h] = curr_alpha.get(h, 0.0) + 0.02 * (hs - lh)
                    curr_beta[a]  = curr_beta.get(a, 0.0)  + 0.02 * (hs - lh)
                    curr_alpha[a] = curr_alpha.get(a, 0.0) + 0.02 * (ascore - la)
                    curr_beta[h]  = curr_beta.get(h, 0.0)  + 0.02 * (ascore - la)
                    
            # Current actual rank
            curr_sorted = sorted(teams, key=lambda t: (actual_pts[t], actual_gd[t], actual_gf[t]), reverse=True)
            curr_rank_map = {t: r for r, t in enumerate(curr_sorted, 1)}
            
            # Remaining fixtures from md+1 to max_md
            rem_fixtures = [m for m in s_matches if m['matchday'] > md]
            rem_by_md = {}
            for m in rem_fixtures:
                rem_by_md.setdefault(m['matchday'], []).append((m['home_team'], m['away_team']))
                
            curr_date_str = md_last_date[md]
            
            if len(rem_fixtures) == 0:
                for rank, t in enumerate(curr_sorted, 1):
                    records.append({
                        'season_id': target_s,
                        'season_name': f"{target_s}/{str(target_s+1)[-2:]}",
                        'matchday': md,
                        'date': curr_date_str,
                        'team': t,
                        'current_rank': rank,
                        'played_matches': actual_played[t],
                        'current_pts': actual_pts[t],
                        'current_gd': actual_gd[t],
                        'exp_final_pts': float(actual_pts[t]),
                        'p_champion': 100.0 if rank == 1 else 0.0,
                        'p_top4': 100.0 if rank <= 4 else 0.0,
                        'p_relegation': 100.0 if rank >= 11 else 0.0
                    })
            else:
                sim_pts = {t: np.zeros(n_sims, dtype=np.int16) for t in teams}
                sim_ranks = {t: np.zeros(n_sims, dtype=np.int8) for t in teams}
                
                total_rem = len(rem_fixtures)
                rnds = np.random.rand(n_sims, total_rem)
                
                for s_i in range(n_sims):
                    s_alpha = dict(curr_alpha)
                    s_beta  = dict(curr_beta)
                    
                    s_pts = dict(actual_pts)
                    s_gd  = dict(actual_gd)
                    cnt = 0
                    
                    for r_md, r_fixtures in rem_by_md.items():
                        lr = (0.02 * 2.5) if (r_md <= 5) else 0.02
                        
                        for h, a in r_fixtures:
                            log_lh = 0.30 + s_alpha.get(h, 0.0) + s_beta.get(a, 0.0) + 0.25
                            log_la = 0.30 + s_alpha.get(a, 0.0) + s_beta.get(h, 0.0)
                            ph, pd_ = get_proba_fast(log_lh, log_la)
                            
                            rnd = rnds[s_i, cnt]
                            cnt += 1
                            
                            if rnd < ph:
                                s_pts[h] += 3
                                s_gd[h] += 1
                                s_gd[a] -= 1
                                err_h, err_a = 0.8, -0.6
                            elif rnd < ph + pd_:
                                s_pts[h] += 1
                                s_pts[a] += 1
                                err_h, err_a = -0.2, -0.2
                            else:
                                s_pts[a] += 3
                                s_gd[a] += 1
                                s_gd[h] -= 1
                                err_h, err_a = -0.6, 0.8
                                
                            s_alpha[h] = s_alpha.get(h, 0.0) + lr * err_h
                            s_beta[a]  = s_beta.get(a, 0.0)  + lr * err_h
                            s_alpha[a] = s_alpha.get(a, 0.0) + lr * err_a
                            s_beta[h]  = s_beta.get(h, 0.0)  + lr * err_a
                            
                    sorted_sim = sorted(teams, key=lambda t: (s_pts[t], s_gd[t]), reverse=True)
                    for r_idx, t in enumerate(sorted_sim, 1):
                        sim_pts[t][s_i] = s_pts[t]
                        sim_ranks[t][s_i] = r_idx
                        
                for t in teams:
                    records.append({
                        'season_id': target_s,
                        'season_name': f"{target_s}/{str(target_s+1)[-2:]}",
                        'matchday': md,
                        'date': curr_date_str,
                        'team': t,
                        'current_rank': curr_rank_map[t],
                        'played_matches': actual_played[t],
                        'current_pts': actual_pts[t],
                        'current_gd': actual_gd[t],
                        'exp_final_pts': round(float(np.mean(sim_pts[t])), 1),
                        'p_champion': round(float(np.mean(sim_ranks[t] == 1) * 100), 1),
                        'p_top4': round(float(np.mean(sim_ranks[t] <= 4) * 100), 1),
                        'p_relegation': round(float(np.mean(sim_ranks[t] >= 11) * 100), 1)
                    })
                    
        # Update global ratings to the end of target_s
        alpha = dict(curr_alpha)
        beta  = dict(curr_beta)

    df_out = pd.DataFrame(records)
    out_path = os.path.join(os.path.dirname(__file__), "nbi_all_seasons_matchday_probabilities.csv")
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] Success! Generated {len(df_out)} state records.")
    print(f"[+] Saved dataset to: {out_path}")

if __name__ == "__main__":
    generate_all_season_data()
