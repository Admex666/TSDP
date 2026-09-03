import os
import math
import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]

def compute_dc_proba(lh, la, rho=-0.10, max_goals=6):
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

def run_matchday_progression(season_target=2024, n_sims=2000):
    df = pd.read_csv(CSV_PATH)
    df['parsed_date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['season_id', 'matchday', 'parsed_date', 'match_id']).reset_index(drop=True)
    
    matches = []
    for r in df.itertuples():
        matches.append({
            'season_id': int(r.season_id),
            'matchday': int(r.matchday),
            'home_team': str(r.home_team),
            'away_team': str(r.away_team),
            'result': str(r.result),
            'home_score': int(r.home_score),
            'away_score': int(r.away_score)
        })
        
    # 1. Warm up Dynamic Dixon-Coles ratings up to target season start
    alpha, beta = {}, {}
    for m in matches:
        if m['season_id'] < season_target:
            h, a = m['home_team'], m['away_team']
            hs, ascore = m['home_score'], m['away_score']
            lh = max(0.1, min(4.5, math.exp(0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25)))
            la = max(0.1, min(4.5, math.exp(0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0))))
            alpha[h] = alpha.get(h, 0.0) + 0.02 * (hs - lh)
            beta[a]  = beta.get(a, 0.0)  + 0.02 * (hs - lh)
            alpha[a] = alpha.get(a, 0.0) + 0.02 * (ascore - la)
            beta[h]  = beta.get(h, 0.0)  + 0.02 * (ascore - la)
            
    season_matches = [m for m in matches if m['season_id'] == season_target]
    teams = sorted(list(set([m['home_team'] for m in season_matches])))
    max_md = max([m['matchday'] for m in season_matches])
    
    # Standings tracker: points, goal_diff, goals_for
    actual_pts = {t: 0 for t in teams}
    actual_gd = {t: 0 for t in teams}
    actual_gf = {t: 0 for t in teams}
    
    progression_records = []
    
    np.random.seed(42)
    
    # Simulate for Matchday 0 (Pre-season) up to Matchday 33
    for md in range(0, max_md + 1):
        if md > 0:
            # Play actual matches of matchday `md`
            md_matches = [m for m in season_matches if m['matchday'] == md]
            for m in md_matches:
                h, a, res, hs, ascore = m['home_team'], m['away_team'], m['result'], m['home_score'], m['away_score']
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
                
                # Update ratings post-match
                lh = max(0.1, min(4.5, math.exp(0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25)))
                la = max(0.1, min(4.5, math.exp(0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0))))
                alpha[h] = alpha.get(h, 0.0) + 0.02 * (hs - lh)
                beta[a]  = beta.get(a, 0.0)  + 0.02 * (hs - lh)
                alpha[a] = alpha.get(a, 0.0) + 0.02 * (ascore - la)
                beta[h]  = beta.get(h, 0.0)  + 0.02 * (ascore - la)

        # Remaining fixtures from matchday `md + 1` to 33
        rem_fixtures = [m for m in season_matches if m['matchday'] > md]
        
        # Precompute probabilities for remaining fixtures with current ratings
        rem_probs = []
        for m in rem_fixtures:
            h, a = m['home_team'], m['away_team']
            log_lh = 0.30 + alpha.get(h, 0.0) + beta.get(a, 0.0) + 0.25
            log_la = 0.30 + alpha.get(a, 0.0) + beta.get(h, 0.0)
            lh = max(0.1, min(4.5, math.exp(log_lh)))
            la = max(0.1, min(4.5, math.exp(log_la)))
            ph, pd_, pa = compute_dc_proba(lh, la, rho=-0.10)
            rem_probs.append((h, a, ph, pd_, pa))
            
        sim_pts = {t: np.zeros(n_sims) for t in teams}
        sim_ranks = {t: np.zeros(n_sims) for t in teams}
        
        if len(rem_probs) == 0:
            # Szezon vége: determinisztikus tabella
            sorted_actual = sorted(teams, key=lambda t: (actual_pts[t], actual_gd[t], actual_gf[t]), reverse=True)
            for rank, t in enumerate(sorted_actual, 1):
                progression_records.append({
                    'matchday': md,
                    'team': t,
                    'current_pts': actual_pts[t],
                    'exp_final_pts': float(actual_pts[t]),
                    'p_champion': 100.0 if rank == 1 else 0.0,
                    'p_top4': 100.0 if rank <= 4 else 0.0,
                    'p_relegation': 100.0 if rank >= 11 else 0.0,
                    'current_rank': rank
                })
        else:
            for s_idx in range(n_sims):
                sim_t_pts = {t: actual_pts[t] for t in teams}
                sim_t_gd  = {t: actual_gd[t] for t in teams}
                
                for h, a, ph, pd_, pa in rem_probs:
                    rnd = np.random.rand()
                    if rnd < ph:
                        sim_t_pts[h] += 3
                        sim_t_gd[h] += 1
                        sim_t_gd[a] -= 1
                    elif rnd < ph + pd_:
                        sim_t_pts[h] += 1
                        sim_t_pts[a] += 1
                    else:
                        sim_t_pts[a] += 3
                        sim_t_gd[a] += 1
                        sim_t_gd[h] -= 1
                        
                sorted_t = sorted(teams, key=lambda t: (sim_t_pts[t], sim_t_gd[t]), reverse=True)
                for rank, t in enumerate(sorted_t, 1):
                    sim_pts[t][s_idx] = sim_t_pts[t]
                    sim_ranks[t][s_idx] = rank
                    
            # Compute current actual rank for display
            curr_sorted = sorted(teams, key=lambda t: (actual_pts[t], actual_gd[t], actual_gf[t]), reverse=True)
            curr_rank_map = {t: r for r, t in enumerate(curr_sorted, 1)}
            
            for t in teams:
                progression_records.append({
                    'matchday': md,
                    'team': t,
                    'current_pts': actual_pts[t],
                    'exp_final_pts': round(float(np.mean(sim_pts[t])), 1),
                    'p_champion': round(float(np.mean(sim_ranks[t] == 1) * 100), 1),
                    'p_top4': round(float(np.mean(sim_ranks[t] <= 4) * 100), 1),
                    'p_relegation': round(float(np.mean(sim_ranks[t] >= 11) * 100), 1),
                    'current_rank': curr_rank_map[t]
                })
                
    df_prog = pd.DataFrame(progression_records)
    out_csv = os.path.join(os.path.dirname(__file__), "nbi_2024_matchday_probabilities.csv")
    df_prog.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Saved full matchday-by-matchday simulations to: {out_csv}")
    
    return df_prog

if __name__ == "__main__":
    df_prog = run_matchday_progression(season_target=2024, n_sims=2000)
    
    # Milestone matchdays: 0 (Rajt), 6 (Szezon eleje), 11 (Első harmad), 17 (Téli szünet/Féltáv), 22 (Kétharmad), 28 (Hajrá), 33 (Vége)
    milestones = [0, 6, 11, 17, 22, 28, 33]
    
    print("\n" + "="*80)
    print("NB I 2024/25 SZEZON FORDULÓNKÉNTI VALÓSZÍNŰSÉG-ALAKULÁS (MILESTONES)")
    print("="*80)
    
    for md in milestones:
        sub = df_prog[df_prog['matchday'] == md].sort_values(by='p_champion', ascending=False)
        if md == 0:
            title = "RAJT ELŐTT (0. FORDULÓ - PRE-SEASON)"
        elif md == 17:
            title = f"{md}. FORDULÓ (TÉLI SZÜNET / FÉLTÁV)"
        elif md == 33:
            title = f"{md}. FORDULÓ (VÉGEREDMÉNY)"
        else:
            title = f"{md}. FORDULÓ UTÁN"
            
        print(f"\n--- {title} ---")
        cols = ['team', 'current_pts', 'exp_final_pts', 'p_champion', 'p_top4', 'p_relegation']
        disp = sub[cols].copy()
        disp.columns = ['Csapat', 'Aktuális Pont', 'Várható Végső Pont', 'Bajnok %', 'Top 4 %', 'Kiesés %']
        print(disp.to_string(index=False))
