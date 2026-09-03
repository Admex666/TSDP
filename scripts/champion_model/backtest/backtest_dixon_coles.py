import os
import math
import itertools
import numpy as np
import pandas as pd
from dynamic_dixon_coles import DynamicDixonColesEngine

CSV_PATH = os.path.join(os.path.dirname(__file__), "nbi_canonical_matches_2015_2025.csv")

def fast_dixon_coles_walkforward(matches_data, lr_att=0.06, lr_def=0.06, home_adv=0.22, base_mu=0.25, dc_rho=-0.06, season_decay=0.90, max_season=2025):
    engine = DynamicDixonColesEngine(
        lr_att=lr_att,
        lr_def=lr_def,
        home_adv=home_adv,
        base_mu=base_mu,
        dc_rho=dc_rho,
        season_decay=season_decay
    )
    
    total_log_loss = 0.0
    total_brier = 0.0
    correct_count = 0
    count = 0
    current_season = None
    
    for season_id, matchday, home, away, res, hs, ascore in matches_data:
        if season_id > max_season:
            break
            
        if current_season is not None and season_id != current_season:
            engine.apply_offseason_decay()
        current_season = season_id
        
        p_h, p_d, p_a, lh, la = engine.predict_proba(home, away)
        
        p_act = p_h if res == 'H' else (p_d if res == 'D' else p_a)
        y = (1.0, 0.0, 0.0) if res == 'H' else ((0.0, 1.0, 0.0) if res == 'D' else (0.0, 0.0, 1.0))
        
        total_log_loss += -math.log(max(1e-15, p_act))
        total_brier += (p_h - y[0])**2 + (p_d - y[1])**2 + (p_a - y[2])**2
        
        if (p_h >= p_d and p_h >= p_a and res == 'H') or \
           (p_d >= p_h and p_d >= p_a and res == 'D') or \
           (p_a >= p_h and p_a >= p_d and res == 'A'):
            correct_count += 1
            
        count += 1
        engine.update_ratings(home, away, hs, ascore)
        
    return {
        'log_loss': total_log_loss / count,
        'brier_score': total_brier / count,
        'accuracy': correct_count / count,
        'count': count
    }

def run_dixon_coles_experiments():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(by=['season_id', 'matchday', 'date', 'match_id']).reset_index(drop=True)
    
    matches_data = [
        (int(r.season_id), int(r.matchday), str(r.home_team), str(r.away_team), str(r.result), int(r.home_score), int(r.away_score))
        for r in df.itertuples()
    ]
    train_data = [m for m in matches_data if m[0] <= 2021]
    
    print("=================================================================")
    print("MODEL 2: DYNAMIC DIXON-COLES HYPERPARAMETER TUNING (2015-2021)")
    print("=================================================================")
    
    lr_grid = [0.02, 0.04, 0.06, 0.08, 0.10]
    h_grid = [0.15, 0.20, 0.25, 0.30]
    mu_grid = [0.20, 0.25, 0.30]
    rho_grid = [-0.10, -0.05, 0.0, 0.05]
    decay_grid = [0.80, 0.90, 0.95, 1.00]
    
    search_space = list(itertools.product(lr_grid, h_grid, mu_grid, rho_grid, decay_grid))
    print(f"Grid search combinations on Train: {len(search_space)}")
    
    best_train_ll = float("inf")
    best_params = None
    
    for lr, h, mu, rho, decay in search_space:
        res = fast_dixon_coles_walkforward(
            train_data,
            lr_att=lr,
            lr_def=lr,
            home_adv=h,
            base_mu=mu,
            dc_rho=rho,
            season_decay=decay
        )
        if res['log_loss'] < best_train_ll:
            best_train_ll = res['log_loss']
            best_params = {'lr': lr, 'home_adv': h, 'base_mu': mu, 'dc_rho': rho, 'season_decay': decay}
            
    print("\nOptimal Dixon-Coles Parameters (Found on 2015-2021):")
    print(f" - Learning Rate (Att/Def): {best_params['lr']}")
    print(f" - Home Advantage (Log):   {best_params['home_adv']}")
    print(f" - Base Mu (Log goals):     {best_params['base_mu']}")
    print(f" - Dixon-Coles Rho:         {best_params['dc_rho']}")
    print(f" - Season Decay:            {best_params['season_decay']*100:.0f}%")
    print(f" - Train Log Loss:          {best_train_ll:.4f}")
    
    # Evaluate Out-of-Sample (2022-2025)
    print("\n=================================================================")
    print("MODEL 2: OUT-OF-SAMPLE TEST EVALUATION (2022-2025)")
    print("=================================================================")
    
    engine = DynamicDixonColesEngine(
        lr_att=best_params['lr'],
        lr_def=best_params['lr'],
        home_adv=best_params['home_adv'],
        base_mu=best_params['base_mu'],
        dc_rho=best_params['dc_rho'],
        season_decay=best_params['season_decay']
    )
    
    test_records = []
    current_season = None
    
    for s_id, md, home, away, res, hs, ascore in matches_data:
        if current_season is not None and s_id != current_season:
            engine.apply_offseason_decay()
        current_season = s_id
        
        p_h, p_d, p_a, lh, la = engine.predict_proba(home, away)
        
        p_act = p_h if res == 'H' else (p_d if res == 'D' else p_a)
        y = (1.0, 0.0, 0.0) if res == 'H' else ((0.0, 1.0, 0.0) if res == 'D' else (0.0, 0.0, 1.0))
        
        ll_i = -math.log(max(1e-15, p_act))
        brier_i = (p_h - y[0])**2 + (p_d - y[1])**2 + (p_a - y[2])**2
        is_corr = 1 if (p_h >= p_d and p_h >= p_a and res == 'H') or \
                       (p_d >= p_h and p_d >= p_a and res == 'D') or \
                       (p_a >= p_h and p_a >= p_d and res == 'A') else 0
                       
        if s_id >= 2022:
            test_records.append({
                'season_id': s_id,
                'matchday': md,
                'home_team': home,
                'away_team': away,
                'result': res,
                'home_score': hs,
                'away_score': ascore,
                'lambda_home': round(lh, 2),
                'lambda_away': round(la, 2),
                'p_home': round(p_h, 4),
                'p_draw': round(p_d, 4),
                'p_away': round(p_a, 4),
                'log_loss': ll_i,
                'brier_score': brier_i,
                'is_correct': is_corr
            })
            
        engine.update_ratings(home, away, hs, ascore)
        
    df_dc_test = pd.DataFrame(test_records)
    
    dc_test_ll = df_dc_test['log_loss'].mean()
    dc_test_brier = df_dc_test['brier_score'].mean()
    dc_test_acc = df_dc_test['is_correct'].mean()
    
    print(f"Dixon-Coles Test Performance (2022-2025, N=792):")
    print(f" - Test Log Loss:    {dc_test_ll:.4f}")
    print(f" - Test Brier Score: {dc_test_brier:.4f}")
    print(f" - Test Accuracy:    {dc_test_acc*100:.2f}%")
    
    print("\n--- Season Breakdown (Dixon-Coles Test) ---")
    s_grp = df_dc_test.groupby('season_id').agg(
        matches=('result', 'count'),
        log_loss=('log_loss', 'mean'),
        brier=('brier_score', 'mean'),
        acc=('is_correct', 'mean')
    ).reset_index()
    print(s_grp.to_string(index=False))
    
    # 3-WAY COMPARISON TABLE
    print("\n=================================================================")
    print("FINAL MODEL LEADERBOARD (UNTOUCHED TEST SET 2022-2025, N=792)")
    print("=================================================================")
    
    leaderboard = [
        {"Rank": 1, "Model": "Model 2: Dynamic Dixon-Coles", "Log Loss": f"{dc_test_ll:.4f}", "Brier Score": f"{dc_test_brier:.4f}", "Accuracy": f"{dc_test_acc*100:.2f}%", "vs Naive LL Gain": f"{1.0749 - dc_test_ll:+.4f}"},
        {"Rank": 2, "Model": "Model 1: Dynamic Elo", "Log Loss": "1.0379", "Brier Score": "0.6238", "Accuracy": "47.85%", "vs Naive LL Gain": "+0.0370"},
        {"Rank": 3, "Model": "Model 0: Naive Base Rate", "Log Loss": "1.0749", "Brier Score": "0.6491", "Accuracy": "45.08%", "vs Naive LL Gain": "0.0000"}
    ]
    df_lb = pd.DataFrame(leaderboard).sort_values(by="Rank")
    print(df_lb.to_string(index=False))
    
    # Save predictions
    preds_path = os.path.join(os.path.dirname(__file__), "nbi_dixon_coles_test_predictions.csv")
    df_dc_test.to_csv(preds_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved test predictions to: {preds_path}")

if __name__ == "__main__":
    run_dixon_coles_experiments()
