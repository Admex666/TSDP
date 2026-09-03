import os
import math
import numpy as np
import pandas as pd

CSV_PATH_HISTORICAL = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0]

def compute_dc_matrix(lh, la, rho=-0.10, max_goals=7):
    m = max_goals + 1
    px = [math.exp(-lh) * (lh**x) / FACT[x] for x in range(m)]
    py = [math.exp(-la) * (la**y) / FACT[y] for y in range(m)]
    
    matrix = np.zeros((m, m))
    tau_00 = 1.0 - lh * la * rho
    tau_01 = 1.0 + lh * rho
    tau_10 = 1.0 + la * rho
    tau_11 = 1.0 - rho
    
    for x in range(m):
        for y in range(m):
            prob = px[x] * py[y]
            if rho != 0.0:
                if x == 0 and y == 0: prob *= tau_00
                elif x == 0 and y == 1: prob *= tau_01
                elif x == 1 and y == 0: prob *= tau_10
                elif x == 1 and y == 1: prob *= tau_11
            matrix[x, y] = max(0.0, prob)
            
    s = matrix.sum()
    if s > 0:
        matrix /= s
    return matrix

def run_comprehensive_audit():
    df = pd.read_csv(CSV_PATH_HISTORICAL)
    df['parsed_date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['season_id', 'matchday', 'parsed_date', 'match_id']).reset_index(drop=True)
    
    # 4 Model Engines
    # Model 1: Static Poisson
    stat_lh = 1.45
    stat_la = 1.15
    stat_matrix = compute_dc_matrix(stat_lh, stat_la, rho=0.0)
    
    # Model 2: Dynamic Poisson (rho=0)
    # Model 3: Dynamic Dixon-Coles (rho=-0.10)
    # Model 4: Dynamic Elo-Implied Poisson
    
    alpha_dc, beta_dc = {}, {}
    alpha_dp, beta_dp = {}, {}
    elo_ratings = {}
    
    results = {
        'Static Poisson': [],
        'Dynamic Elo-Implied': [],
        'Dynamic Poisson': [],
        'Dynamic Dixon-Coles': []
    }
    
    all_score_events = [] # For global scoreline calibration
    
    for r in df.itertuples():
        s_id = int(r.season_id)
        h, a = str(r.home_team), str(r.away_team)
        hs, ascore = int(r.home_score), int(r.away_score)
        res = str(r.result)
        is_test = (s_id >= 2022)
        
        # 1. Predictions BEFORE match
        # --- A. Static Poisson ---
        mat_stat = stat_matrix
        
        # --- B. Dynamic Poisson (rho = 0.0) ---
        log_lh_dp = 0.30 + alpha_dp.get(h, 0.0) + beta_dp.get(a, 0.0) + 0.25
        log_la_dp = 0.30 + alpha_dp.get(a, 0.0) + beta_dp.get(h, 0.0)
        lh_dp = max(0.1, min(5.0, math.exp(log_lh_dp)))
        la_dp = max(0.1, min(5.0, math.exp(log_la_dp)))
        mat_dp = compute_dc_matrix(lh_dp, la_dp, rho=0.0)
        
        # --- C. Dynamic Dixon-Coles (rho = -0.10) ---
        log_lh_dc = 0.30 + alpha_dc.get(h, 0.0) + beta_dc.get(a, 0.0) + 0.25
        log_la_dc = 0.30 + alpha_dc.get(a, 0.0) + beta_dc.get(h, 0.0)
        lh_dc = max(0.1, min(5.0, math.exp(log_lh_dc)))
        la_dc = max(0.1, min(5.0, math.exp(log_la_dc)))
        mat_dc = compute_dc_matrix(lh_dc, la_dc, rho=-0.10)
        
        # --- D. Dynamic Elo-Implied Poisson ---
        rh = elo_ratings.get(h, 1500.0)
        ra = elo_ratings.get(a, 1500.0)
        d_elo = (rh + 60.0) - ra
        # Elo to expected goals conversion: base 2.6 goals total
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        tot_g = 2.60
        lh_elo = max(0.1, min(5.0, tot_g * e_h))
        la_elo = max(0.1, min(5.0, tot_g * (1.0 - e_h)))
        mat_elo = compute_dc_matrix(lh_elo, la_elo, rho=0.0)
        
        models_mat = {
            'Static Poisson': (mat_stat, lh_stat_tot:=stat_lh+stat_la),
            'Dynamic Elo-Implied': (mat_elo, lh_elo+la_elo),
            'Dynamic Poisson': (mat_dp, lh_dp+la_dp),
            'Dynamic Dixon-Coles': (mat_dc, lh_dc+la_dc)
        }
        
        if is_test:
            # Evaluate test match
            # Clamped actual score coordinates for matrix index
            cx = min(7, hs)
            cy = min(7, ascore)
            actual_tg = hs + ascore
            
            for m_name, (mat, exp_tg) in models_mat.items():
                p_actual_score = max(1e-6, mat[cx, cy])
                ll_score = -math.log(p_actual_score)
                
                # 1X2 Probabilities
                p_h = float(np.tril(mat, -1).sum())
                p_d = float(np.trace(mat))
                p_a = float(np.triu(mat, 1).sum())
                p_res = p_h if res == 'H' else (p_d if res == 'D' else p_a)
                ll_1x2 = -math.log(max(1e-15, p_res))
                
                # Total goals distribution PMF from matrix
                tg_pmf = [0.0] * 16
                for x in range(8):
                    for y in range(8):
                        tg_pmf[x + y] += mat[x, y]
                        
                # Over / Under markets (0.5 .. 5.5)
                ou_lls = {}
                ou_briers = {}
                for th in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    floor_k = int(th + 0.5)
                    p_over = sum(tg_pmf[k] for k in range(floor_k, len(tg_pmf)))
                    act_over = 1 if actual_tg >= floor_k else 0
                    p_act = p_over if act_over == 1 else (1.0 - p_over)
                    ou_lls[th] = -math.log(max(1e-15, p_act))
                    ou_briers[th] = (p_over - act_over) ** 2
                    
                # BTTS
                p_btts = float(mat[1:, 1:].sum())
                act_btts = 1 if (hs > 0 and ascore > 0) else 0
                p_act_btts = p_btts if act_btts == 1 else (1.0 - p_btts)
                ll_btts = -math.log(max(1e-15, p_act_btts))
                brier_btts = (p_btts - act_btts) ** 2
                
                # Exact score hits
                max_idx = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
                is_exact_top1 = 1 if (max_idx[0] == hs and max_idx[1] == ascore) else 0
                
                flat_sorted = np.argsort(-mat.ravel())[:3]
                top3_scores = [(idx // 8, idx % 8) for idx in flat_sorted]
                is_exact_top3 = 1 if (hs, ascore) in top3_scores else 0
                
                rec = {
                    'season_id': s_id,
                    'match_id': r.match_id,
                    'll_scoreline': ll_score,
                    'p_actual_score': p_actual_score,
                    'll_1x2': ll_1x2,
                    'll_btts': ll_btts,
                    'brier_btts': brier_btts,
                    'is_exact_top1': is_exact_top1,
                    'is_exact_top3': is_exact_top3,
                    'actual_tg': actual_tg
                }
                for th in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    rec[f'll_ou_{th}'] = ou_lls[th]
                    rec[f'brier_ou_{th}'] = ou_briers[th]
                    rec[f'p_ou_{th}'] = sum(tg_pmf[k] for k in range(int(th+0.5), len(tg_pmf)))
                    rec[f'act_ou_{th}'] = 1 if actual_tg >= int(th+0.5) else 0
                results[m_name].append(rec)
                
            # Track Dixon-Coles predicted probabilities for scoreline calibration
            for x in range(5):
                for y in range(5):
                    all_score_events.append({
                        'score': f"{x}-{y}",
                        'pred_p': mat_dc[x, y],
                        'actual_occurred': 1 if (hs == x and ascore == y) else 0
                    })

        # 2. Update models post-match
        # Update DP
        err_h_dp = float(hs) - lh_dp
        err_a_dp = float(ascore) - la_dp
        alpha_dp[h] = alpha_dp.get(h, 0.0) + 0.02 * err_h_dp
        beta_dp[a]  = beta_dp.get(a, 0.0)  + 0.02 * err_h_dp
        alpha_dp[a] = alpha_dp.get(a, 0.0) + 0.02 * err_a_dp
        beta_dp[h]  = beta_dp.get(h, 0.0)  + 0.02 * err_a_dp
        
        # Update DC
        err_h_dc = float(hs) - lh_dc
        err_a_dc = float(ascore) - la_dc
        alpha_dc[h] = alpha_dc.get(h, 0.0) + 0.02 * err_h_dc
        beta_dc[a]  = beta_dc.get(a, 0.0)  + 0.02 * err_h_dc
        alpha_dc[a] = alpha_dc.get(a, 0.0) + 0.02 * err_a_dc
        beta_dc[h]  = beta_dc.get(h, 0.0)  + 0.02 * err_a_dc
        
        # Update Elo
        s_act = 1.0 if res == 'H' else (0.5 if res == 'D' else 0.0)
        e_act = 1.0 / (1.0 + math.pow(10.0, -((rh + 60.0) - ra) / 400.0))
        elo_ratings[h] = rh + 15.0 * (s_act - e_act)
        elo_ratings[a] = ra - 15.0 * (s_act - e_act)
        
    # Build Evaluation DataFrames
    print("================================================================================")
    print("V3 COMPREHENSIVE SCORELINE & PROBABILISTIC MARKET AUDIT (TEST 2022-2025, N=792)")
    print("================================================================================")
    
    # 1. Overall Scoreline Log Loss Benchmark
    print("\n--- 1. FULL SCORELINE LOG LOSS BENCHMARK (Proper Scoring Rule over Full 2D Grid) ---")
    score_bench = []
    for m_name in ['Static Poisson', 'Dynamic Elo-Implied', 'Dynamic Poisson', 'Dynamic Dixon-Coles']:
        m_df = pd.DataFrame(results[m_name])
        score_bench.append({
            'Modell': m_name,
            'Full Scoreline Log Loss': f"{m_df['ll_scoreline'].mean():.4f}",
            '1X2 Match Log Loss': f"{m_df['ll_1x2'].mean():.4f}",
            'Exact Score Top-1 Hit Rate': f"{m_df['is_exact_top1'].mean()*100:.2f}%",
            'Exact Score Top-3 Hit Rate': f"{m_df['is_exact_top3'].mean()*100:.2f}%",
            'BTTS Log Loss': f"{m_df['ll_btts'].mean():.4f}"
        })
    print(pd.DataFrame(score_bench).to_string(index=False))
    
    # 2. Total Goals Multi-Threshold Probabilistic Score
    print("\n--- 2. MULTI-THRESHOLD OVER / UNDER PROBABILISTIC EVALUATION (Dynamic Dixon-Coles) ---")
    dc_df = pd.DataFrame(results['Dynamic Dixon-Coles'])
    ou_summary = []
    for th in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        ll = dc_df[f'll_ou_{th}'].mean()
        brier = dc_df[f'brier_ou_{th}'].mean()
        pred_p = dc_df[f'p_ou_{th}'].mean() * 100
        act_p = dc_df[f'act_ou_{th}'].mean() * 100
        acc = ((dc_df[f'p_ou_{th}'] > 0.5) == dc_df[f'act_ou_{th}']).mean() * 100
        ou_summary.append({
            'Piac (Threshold)': f"Over / Under {th}",
            'Log Loss': f"{ll:.4f}",
            'Brier Score': f"{brier:.4f}",
            'Accuracy': f"{acc:.2f}%",
            'Predikalt Over %': f"{pred_p:.1f}%",
            'Valos Over %': f"{act_p:.1f}%",
            'Kalibracios Elteres': f"{act_p - pred_p:+.2f}%"
        })
    print(pd.DataFrame(ou_summary).to_string(index=False))
    
    # 3. Scoreline Probability Calibration (Reliability Curve across Predicted Probability Bins)
    print("\n--- 3. SCORELINE PROBABILITY CALIBRATION (Binned Reliability across all Scorelines) ---")
    df_cal = pd.DataFrame(all_score_events)
    bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0]
    df_cal['p_bin'] = pd.cut(df_cal['pred_p'], bins=bins)
    
    cal_table = []
    for interval, grp in df_cal.groupby('p_bin', observed=False):
        if len(grp) == 0: continue
        mean_pred = grp['pred_p'].mean() * 100
        mean_act = grp['actual_occurred'].mean() * 100
        cal_table.append({
            'Predikalt P(Score) Sav': f"{interval.left*100:.0f}% - {interval.right*100:.0f}%",
            'Megfigyelesek': len(grp),
            'Atlagos Predikcio': f"{mean_pred:.2f}%",
            'Valos Gyakorisag': f"{mean_act:.2f}%",
            'Kalibracios Elteres': f"{mean_act - mean_pred:+.2f}%"
        })
    print(pd.DataFrame(cal_table).to_string(index=False))
    
    # 4. Key Scorelines Individual Calibration
    print("\n--- 4. TOP 8 INDIVIDUAL SCORELINE CALIBRATION ---")
    top_scores = ['1-1', '2-1', '1-0', '1-2', '0-0', '2-0', '0-1', '2-2']
    indiv_rows = []
    for sc in top_scores:
        sc_df = df_cal[df_cal['score'] == sc]
        p_pred = sc_df['pred_p'].mean() * 100
        p_act = sc_df['actual_occurred'].mean() * 100
        indiv_rows.append({
            'Pontos Eredmeny': sc,
            'Meccsek': len(sc_df),
            'Predikalt Atlag %': f"{p_pred:.2f}%",
            'Valos Gyakorisag %': f"{p_act:.2f}%",
            'Elteres': f"{p_act - p_pred:+.2f}%"
        })
    print(pd.DataFrame(indiv_rows).to_string(index=False))

if __name__ == "__main__":
    run_comprehensive_audit()
