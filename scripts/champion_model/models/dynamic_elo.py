import math
import numpy as np
import pandas as pd

class DynamicEloEngine:
    def __init__(self, k_factor=20.0, home_advantage=60.0, season_regression=0.85, default_elo=1500.0, draw_param=0.26):
        """
        Dynamic Elo Engine for Football (3-Way Outcomes).
        
        Parameters:
        - k_factor (float): K-factor weight for post-match rating update.
        - home_advantage (float): Elo bonus for the home team.
        - season_regression (float): Off-season mean regression factor (0.0 to 1.0).
        - default_elo (float): Default Elo rating for new/unseen teams (1500.0).
        - draw_param (float): Base probability of a draw at D_elo = 0 (typically ~0.26 in football).
        """
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.season_regression = float(season_regression)
        self.default_elo = float(default_elo)
        self.draw_param = float(draw_param)
        
        # Team ratings dictionary
        self.ratings = {}
        
    def get_rating(self, team_name):
        """Returns current Elo rating of a team (defaults to default_elo)."""
        return self.ratings.get(team_name, self.default_elo)
        
    def set_rating(self, team_name, elo_value):
        """Manually sets Elo rating of a team."""
        self.ratings[team_name] = float(elo_value)

    def get_effective_ratings(self, home_team, away_team):
        """Returns effective home rating (R_home + H) and away rating (R_away)."""
        r_home = self.get_rating(home_team) + self.home_advantage
        r_away = self.get_rating(away_team)
        return r_home, r_away

    def predict_proba(self, home_team, away_team):
        """
        Calculates 3-way probabilities P(Home), P(Draw), P(Away) using rating difference.
        
        D_elo = (R_home + H) - R_away
        Expected home score E_H = 1 / (1 + 10^(-D_elo / 400))
        """
        r_home_eff, r_away = self.get_effective_ratings(home_team, away_team)
        d_elo = r_home_eff - r_away
        
        # Expected Home Score E_H from classical Logistic Elo
        e_h = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        
        # Draw probability model: P(Draw) decays smoothly as |D_elo| grows
        # P(D) = draw_param * exp(-(d_elo / 350)^2)
        p_draw = self.draw_param * math.exp(-math.pow(d_elo / 350.0, 2))
        
        # Ensure P(H) and P(A) align with E_H = P(H) + 0.5 * P(D)
        p_home = e_h - 0.5 * p_draw
        p_away = (1.0 - e_h) - 0.5 * p_draw
        
        # Clip to valid range [0.001, 0.998] and normalize
        p_home = max(0.001, p_home)
        p_draw = max(0.001, p_draw)
        p_away = max(0.001, p_away)
        
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total
        
        return p_home, p_draw, p_away

    def update_ratings(self, home_team, away_team, actual_result):
        """
        Updates ratings post-match based on actual result.
        actual_result: 'H' (Home win = 1.0), 'D' (Draw = 0.5), 'A' (Away win = 0.0)
        """
        r_home_old = self.get_rating(home_team)
        r_away_old = self.get_rating(away_team)
        
        # Effective ratings for expected score calculation
        r_home_eff = r_home_old + self.home_advantage
        d_elo = r_home_eff - r_away_old
        e_home = 1.0 / (1.0 + math.pow(10.0, -d_elo / 400.0))
        
        # Convert actual result to numerical score
        if actual_result == 'H':
            s_actual = 1.0
        elif actual_result == 'D':
            s_actual = 0.5
        elif actual_result == 'A':
            s_actual = 0.0
        else:
            raise ValueError(f"Invalid actual_result '{actual_result}'. Expected 'H', 'D', or 'A'.")
            
        # Rating delta
        delta = self.k_factor * (s_actual - e_home)
        
        r_home_new = r_home_old + delta
        r_away_new = r_away_old - delta
        
        self.ratings[home_team] = r_home_new
        self.ratings[away_team] = r_away_new
        
        return r_home_new, r_away_new, delta

    def apply_offseason_regression(self):
        """
        Regresses all active team ratings towards the default mean (1500) between seasons.
        R_new = 1500 + rho * (R_old - 1500)
        """
        for team in list(self.ratings.keys()):
            old_rating = self.ratings[team]
            new_rating = self.default_elo + self.season_regression * (old_rating - self.default_elo)
            self.ratings[team] = new_rating
