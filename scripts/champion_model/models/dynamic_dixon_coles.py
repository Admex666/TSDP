import math
import numpy as np

# Precomputed factorials for k in 0..7
FACT = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0]

class DynamicDixonColesEngine:
    def __init__(self, lr_att=0.06, lr_def=0.06, home_adv=0.22, base_mu=0.25, dc_rho=-0.06, season_decay=0.90, max_goals=6):
        self.lr_att = float(lr_att)
        self.lr_def = float(lr_def)
        self.home_adv = float(home_adv)
        self.base_mu = float(base_mu)
        self.dc_rho = float(dc_rho)
        self.season_decay = float(season_decay)
        self.max_goals = int(max_goals)
        
        self.alpha = {}
        self.beta = {}
        
    def get_attack(self, team):
        return self.alpha.get(team, 0.0)
        
    def get_defense(self, team):
        return self.beta.get(team, 0.0)
        
    def predict_expected_goals(self, home_team, away_team):
        att_h = self.get_attack(home_team)
        def_a = self.get_defense(away_team)
        att_a = self.get_attack(away_team)
        def_h = self.get_defense(home_team)
        
        log_lh = self.base_mu + att_h + def_a + self.home_adv
        log_la = self.base_mu + att_a + def_h
        
        lh = max(0.1, min(4.5, math.exp(log_lh)))
        la = max(0.1, min(4.5, math.exp(log_la)))
        return lh, la

    def predict_proba(self, home_team, away_team):
        lh, la = self.predict_expected_goals(home_team, away_team)
        
        m = self.max_goals + 1
        px = [math.exp(-lh) * (lh**x) / FACT[x] for x in range(m)]
        py = [math.exp(-la) * (la**y) / FACT[y] for y in range(m)]
        
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        
        # 0-0, 0-1, 1-0, 1-1 with Dixon-Coles tau adjustment
        tau_00 = 1.0 - lh * la * self.dc_rho
        tau_01 = 1.0 + lh * self.dc_rho
        tau_10 = 1.0 + la * self.dc_rho
        tau_11 = 1.0 - self.dc_rho
        
        for x in range(m):
            px_val = px[x]
            for y in range(m):
                prob = px_val * py[y]
                if x == 0 and y == 0: prob *= tau_00
                elif x == 0 and y == 1: prob *= tau_01
                elif x == 1 and y == 0: prob *= tau_10
                elif x == 1 and y == 1: prob *= tau_11
                
                if x > y: p_home += prob
                elif x == y: p_draw += prob
                else: p_away += prob
                
        tot = p_home + p_draw + p_away
        p_home = max(0.001, p_home / tot)
        p_draw = max(0.001, p_draw / tot)
        p_away = max(0.001, p_away / tot)
        tot2 = p_home + p_draw + p_away
        
        return p_home / tot2, p_draw / tot2, p_away / tot2, lh, la

    def update_ratings(self, home_team, away_team, home_score, away_score):
        lh, la = self.predict_expected_goals(home_team, away_team)
        
        err_h = float(home_score) - lh
        err_a = float(away_score) - la
        
        self.alpha[home_team] = self.get_attack(home_team) + self.lr_att * err_h
        self.beta[away_team]  = self.get_defense(away_team) + self.lr_def * err_h
        self.alpha[away_team] = self.get_attack(away_team) + self.lr_att * err_a
        self.beta[home_team]  = self.get_defense(home_team) + self.lr_def * err_a

    def apply_offseason_decay(self):
        for t in list(self.alpha.keys()):
            self.alpha[t] *= self.season_decay
            self.beta[t] *= self.season_decay
