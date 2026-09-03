import os
import math
import numpy as np
import pandas as pd

CSV_PATH_HISTORICAL = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0]

class FullScorelineEngine:
    """
    V3 Full Scoreline Probability Engine for Football.
    Computes the complete joint probability matrix P(X=x, Y=y) under Dynamic Dixon-Coles,
    and derives exact scorelines, 3-way outcomes, Over/Under markets, BTTS, and expected values.
    """
    def __init__(self, lr_att=0.02, lr_def=0.02, home_adv=0.25, base_mu=0.30, dc_rho=-0.10, max_goals=7):
        self.lr_att = float(lr_att)
        self.lr_def = float(lr_def)
        self.home_adv = float(home_adv)
        self.base_mu = float(base_mu)
        self.dc_rho = float(dc_rho)
        self.max_goals = int(max_goals)
        
        self.alpha = {} # Attack strength
        self.beta = {}  # Defensive vulnerability (positive = concedes more)
        
    def get_attack(self, team):
        return self.alpha.get(team, 0.0)
        
    def get_defense(self, team):
        return self.beta.get(team, 0.0)

    def predict_match_full_distribution(self, home_team, away_team):
        """
        Computes all market probabilities and the full scoreline matrix for home_team vs away_team.
        """
        att_h = self.get_attack(home_team)
        def_a = self.get_defense(away_team)
        att_a = self.get_attack(away_team)
        def_h = self.get_defense(home_team)
        
        log_lh = self.base_mu + att_h + def_a + self.home_adv
        log_la = self.base_mu + att_a + def_h
        
        lh = max(0.1, min(5.0, math.exp(log_lh)))
        la = max(0.1, min(5.0, math.exp(log_la)))
        
        m = self.max_goals + 1
        px = [math.exp(-lh) * (lh**x) / FACT[x] for x in range(m)]
        py = [math.exp(-la) * (la**y) / FACT[y] for y in range(m)]
        
        # Build 2D joint score matrix
        matrix = np.zeros((m, m))
        tau_00 = 1.0 - lh * la * self.dc_rho
        tau_01 = 1.0 + lh * self.dc_rho
        tau_10 = 1.0 + la * self.dc_rho
        tau_11 = 1.0 - self.dc_rho
        
        for x in range(m):
            for y in range(m):
                prob = px[x] * py[y]
                if x == 0 and y == 0: prob *= tau_00
                elif x == 0 and y == 1: prob *= tau_01
                elif x == 1 and y == 0: prob *= tau_10
                elif x == 1 and y == 1: prob *= tau_11
                matrix[x, y] = max(0.0, prob)
                
        matrix /= matrix.sum()
        
        # 1. 3-Way Outcomes
        p_home = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, 1).sum())
        
        # 2. Total Goals distribution
        total_goals_pmf = {}
        for g in range(0, (2 * self.max_goals) + 1):
            total_goals_pmf[g] = 0.0
            
        for x in range(m):
            for y in range(m):
                total_goals_pmf[x + y] += matrix[x, y]
                
        # 3. Over / Under Markets
        p_over_0_5 = sum(total_goals_pmf[g] for g in range(1, len(total_goals_pmf)))
        p_over_1_5 = sum(total_goals_pmf[g] for g in range(2, len(total_goals_pmf)))
        p_over_2_5 = sum(total_goals_pmf[g] for g in range(3, len(total_goals_pmf)))
        p_over_3_5 = sum(total_goals_pmf[g] for g in range(4, len(total_goals_pmf)))
        
        p_under_0_5 = 1.0 - p_over_0_5
        p_under_1_5 = 1.0 - p_over_1_5
        p_under_2_5 = 1.0 - p_over_2_5
        p_under_3_5 = 1.0 - p_over_3_5
        
        # 4. Both Teams to Score (BTTS)
        p_btts_yes = float(matrix[1:, 1:].sum())
        p_btts_no  = 1.0 - p_btts_yes
        
        # 5. Most Likely Score
        max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        most_likely_score = f"{max_idx[0]}:{max_idx[1]}"
        most_likely_score_prob = float(matrix[max_idx])
        
        # Top 5 most likely scores
        flat_sorted = np.argsort(-matrix.ravel())
        top_scores = []
        for idx in flat_sorted[:5]:
            x = idx // m
            y = idx % m
            top_scores.append((f"{x}:{y}", float(matrix[x, y])))
            
        return {
            'lambda_home': round(lh, 3),
            'lambda_away': round(la, 3),
            'expected_total_goals': round(lh + la, 3),
            'expected_goal_diff': round(lh - la, 3),
            'p_home': round(p_home, 4),
            'p_draw': round(p_draw, 4),
            'p_away': round(p_away, 4),
            # Key Exact Scores
            'p_0_0': round(float(matrix[0, 0]), 4),
            'p_1_0': round(float(matrix[1, 0]), 4),
            'p_0_1': round(float(matrix[0, 1]), 4),
            'p_1_1': round(float(matrix[1, 1]), 4),
            'p_2_0': round(float(matrix[2, 0]), 4),
            'p_2_1': round(float(matrix[2, 1]), 4),
            'p_1_2': round(float(matrix[1, 2]), 4),
            'p_2_2': round(float(matrix[2, 2]), 4),
            'p_3_0': round(float(matrix[3, 0]), 4),
            'p_3_1': round(float(matrix[3, 1]), 4),
            'p_3_2': round(float(matrix[3, 2]), 4),
            'p_0_2': round(float(matrix[0, 2]), 4),
            'p_0_3': round(float(matrix[0, 3]), 4),
            'p_1_3': round(float(matrix[1, 3]), 4),
            'p_2_3': round(float(matrix[2, 3]), 4),
            # Over / Under
            'p_over_0_5': round(p_over_0_5, 4),
            'p_under_0_5': round(p_under_0_5, 4),
            'p_over_1_5': round(p_over_1_5, 4),
            'p_under_1_5': round(p_under_1_5, 4),
            'p_over_2_5': round(p_over_2_5, 4),
            'p_under_2_5': round(p_under_2_5, 4),
            'p_over_3_5': round(p_over_3_5, 4),
            'p_under_3_5': round(p_under_3_5, 4),
            # BTTS
            'p_btts_yes': round(p_btts_yes, 4),
            'p_btts_no': round(p_btts_no, 4),
            # Most likely
            'most_likely_score': most_likely_score,
            'most_likely_score_prob': round(most_likely_score_prob, 4),
            'top_scores': top_scores,
            'matrix': matrix,
            'total_goals_pmf': total_goals_pmf
        }

    def update_ratings(self, home_team, away_team, home_score, away_score):
        """
        Gradient update on Attack (alpha) and Defense vulnerability (beta):
        e_H = Goals_H - lambda_H
        e_A = Goals_A - lambda_A
        """
        att_h = self.get_attack(home_team)
        def_a = self.get_defense(away_team)
        att_a = self.get_attack(away_team)
        def_h = self.get_defense(home_team)
        
        lh = max(0.1, min(5.0, math.exp(self.base_mu + att_h + def_a + self.home_adv)))
        la = max(0.1, min(5.0, math.exp(self.base_mu + att_a + def_h)))
        
        err_h = float(home_score) - lh
        err_a = float(away_score) - la
        
        self.alpha[home_team] = att_h + self.lr_att * err_h
        self.beta[away_team]  = def_a + self.lr_def * err_h
        self.alpha[away_team] = att_a + self.lr_att * err_a
        self.beta[home_team]  = def_h + self.lr_def * err_a

def run_scoreline_backtest_and_calibration():
    """
    Runs walk-forward backtest and computes full goal-distribution & market calibration.
    """
    df = pd.read_csv(CSV_PATH_HISTORICAL)
    df['parsed_date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['season_id', 'matchday', 'parsed_date', 'match_id']).reset_index(drop=True)
    
    engine = FullScorelineEngine(
        lr_att=0.02,
        lr_def=0.02,
        home_adv=0.25,
        base_mu=0.30,
        dc_rho=-0.10
    )
    
    match_records = []
    
    for r in df.itertuples():
        s_id = int(r.season_id)
        h, a = str(r.home_team), str(r.away_team)
        hs, ascore = int(r.home_score), int(r.away_score)
        res = str(r.result)
        
        # 1. Predict full distribution prior to match
        dist = engine.predict_match_full_distribution(h, a)
        
        actual_total_goals = hs + ascore
        actual_is_over_2_5 = 1 if actual_total_goals >= 3 else 0
        actual_is_btts_yes = 1 if (hs > 0 and ascore > 0) else 0
        actual_score_str = f"{hs}:{ascore}"
        
        # Log losses for markets
        ll_hda = -math.log(max(1e-15, dist['p_home'] if res == 'H' else (dist['p_draw'] if res == 'D' else dist['p_away'])))
        ll_ou25 = -math.log(max(1e-15, dist['p_over_2_5'] if actual_is_over_2_5 else dist['p_under_2_5']))
        ll_btts = -math.log(max(1e-15, dist['p_btts_yes'] if actual_is_btts_yes else dist['p_btts_no']))
        
        is_exact_score_hit = 1 if dist['most_likely_score'] == actual_score_str else 0
        is_top3_score_hit  = 1 if actual_score_str in [s[0] for s in dist['top_scores'][:3]] else 0
        
        rec = {
            'season_id': s_id,
            'matchday': int(r.matchday),
            'date': str(r.date),
            'home_team': h,
            'away_team': a,
            'actual_home_score': hs,
            'actual_away_score': ascore,
            'actual_result': res,
            'actual_total_goals': actual_total_goals,
            'actual_is_over_2_5': actual_is_over_2_5,
            'actual_is_btts_yes': actual_is_btts_yes,
            'actual_score': actual_score_str,
            # Engine Predictions
            'lambda_home': dist['lambda_home'],
            'lambda_away': dist['lambda_away'],
            'expected_total_goals': dist['expected_total_goals'],
            'p_home': dist['p_home'],
            'p_draw': dist['p_draw'],
            'p_away': dist['p_away'],
            'p_over_1_5': dist['p_over_1_5'],
            'p_over_2_5': dist['p_over_2_5'],
            'p_over_3_5': dist['p_over_3_5'],
            'p_btts_yes': dist['p_btts_yes'],
            'most_likely_score': dist['most_likely_score'],
            'most_likely_score_prob': dist['most_likely_score_prob'],
            # Key Scoreline Probabilities
            'p_0_0': dist['p_0_0'],
            'p_1_0': dist['p_1_0'],
            'p_0_1': dist['p_0_1'],
            'p_1_1': dist['p_1_1'],
            'p_2_0': dist['p_2_0'],
            'p_2_1': dist['p_2_1'],
            'p_1_2': dist['p_1_2'],
            'p_2_2': dist['p_2_2'],
            # Evaluation
            'll_hda': ll_hda,
            'll_ou25': ll_ou25,
            'll_btts': ll_btts,
            'is_exact_score_hit': is_exact_score_hit,
            'is_top3_score_hit': is_top3_score_hit
        }
        
        # Save individual total goals PMF for calibration
        for g in range(6):
            rec[f'pred_pmf_tg_{g}'] = dist['total_goals_pmf'].get(g, 0.0)
        rec['pred_pmf_tg_6_plus'] = sum(dist['total_goals_pmf'].get(g, 0.0) for g in range(6, 15))
        
        match_records.append(rec)
        
        # 2. Update ratings post match
        engine.update_ratings(h, a, hs, ascore)
        
    df_all = pd.DataFrame(match_records)
    
    # Save full predictions CSV
    out_csv = os.path.join(os.path.dirname(__file__), "nbi_full_scoreline_predictions_2015_2025.csv")
    df_all.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[+] Saved full scoreline predictions to: {out_csv}")
    
    # Evaluate Untouched Test Set (2022-2025, N=792)
    test_df = df_all[df_all['season_id'] >= 2022].copy()
    
    print("\n=================================================================")
    print("V3 FULL SCORELINE & MARKET CALIBRATION REPORT (TEST SET 2022-2025, N=792)")
    print("=================================================================")
    
    print("\n--- 1. OVERALL MARKET PERFORMANCE METRICS ---")
    print(f"H/D/A Match Log Loss:        {test_df['ll_hda'].mean():.4f}")
    print(f"Over / Under 2.5 Log Loss:   {test_df['ll_ou25'].mean():.4f} (Accuracy: {((test_df['p_over_2_5'] > 0.5) == test_df['actual_is_over_2_5']).mean()*100:.2f}%)")
    print(f"BTTS Log Loss:               {test_df['ll_btts'].mean():.4f} (Accuracy: {((test_df['p_btts_yes'] > 0.5) == test_df['actual_is_btts_yes']).mean()*100:.2f}%)")
    print(f"Exact Score Top-1 Hit Rate:  {test_df['is_exact_score_hit'].mean()*100:.2f}%")
    print(f"Exact Score Top-3 Hit Rate:  {test_df['is_top3_score_hit'].mean()*100:.2f}%")
    
    print("\n--- 2. GÓLSZÁM-ELOSZLÁS KALIBRÁCIÓ (TOTAL GOALS DISTRIBUTION) ---")
    tg_rows = []
    for g in range(6):
        pred_pct = test_df[f'pred_pmf_tg_{g}'].mean() * 100
        act_pct = (test_df['actual_total_goals'] == g).mean() * 100
        tg_rows.append({
            'Gólszám (Total Goals)': f"{g} gól",
            'Predikált Átlag %': f"{pred_pct:.2f}%",
            'Valós Történelmi %': f"{act_pct:.2f}%",
            'Kalibrációs Eltérés': f"{act_pct - pred_pct:+.2f}%"
        })
    pred_6p = test_df['pred_pmf_tg_6_plus'].mean() * 100
    act_6p = (test_df['actual_total_goals'] >= 6).mean() * 100
    tg_rows.append({
        'Gólszám (Total Goals)': "6+ gól",
        'Predikált Átlag %': f"{pred_6p:.2f}%",
        'Valós Történelmi %': f"{act_6p:.2f}%",
        'Kalibrációs Eltérés': f"{act_6p - pred_6p:+.2f}%"
    })
    print(pd.DataFrame(tg_rows).to_string(index=False))
    
    print("\n--- 3. OVER / UNDER 2.5 MEGBÍZHATÓSÁGI KALIBRÁCIÓ ---")
    ou_bins = [0.0, 0.40, 0.48, 0.52, 0.60, 1.0]
    test_df['ou_bin'] = pd.cut(test_df['p_over_2_5'], bins=ou_bins)
    ou_rows = []
    for interval, grp in test_df.groupby('ou_bin', observed=False):
        if len(grp) == 0: continue
        p_mean = grp['p_over_2_5'].mean() * 100
        act_rate = grp['actual_is_over_2_5'].mean() * 100
        ou_rows.append({
            'P(Over 2.5) Sáv': str(interval),
            'Meccsek': len(grp),
            'Predikált Átlag': f"{p_mean:.1f}%",
            'Valós Over 2.5 %': f"{act_rate:.1f}%",
            'Eltérés': f"{act_rate - p_mean:+.1f}%"
        })
    print(pd.DataFrame(ou_rows).to_string(index=False))

    print("\n--- 4. BTTS (MIND KÉT CSAPAT GÓLT SZEREZ) KALIBRÁCIÓ ---")
    btts_bins = [0.0, 0.45, 0.52, 0.58, 0.65, 1.0]
    test_df['btts_bin'] = pd.cut(test_df['p_btts_yes'], bins=btts_bins)
    btts_rows = []
    for interval, grp in test_df.groupby('btts_bin', observed=False):
        if len(grp) == 0: continue
        p_mean = grp['p_btts_yes'].mean() * 100
        act_rate = grp['actual_is_btts_yes'].mean() * 100
        btts_rows.append({
            'P(BTTS Yes) Sáv': str(interval),
            'Meccsek': len(grp),
            'Predikált Átlag': f"{p_mean:.1f}%",
            'Valós BTTS %': f"{act_rate:.1f}%",
            'Eltérés': f"{act_rate - p_mean:+.1f}%"
        })
    print(pd.DataFrame(btts_rows).to_string(index=False))

if __name__ == "__main__":
    run_scoreline_backtest_and_calibration()
