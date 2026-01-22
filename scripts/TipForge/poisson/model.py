import pandas as pd
import numpy as np
from scipy.stats import poisson

class PoissonModel:
    def __init__(self):
        self.team_stats = None
        self.avg_home_goals = 0
        self.avg_away_goals = 0

    def fit(self, df, prediction_date, time_decay=0.0):
        """
        Calculates attack and defense strengths for each team based on the provided dataframe.
        If time_decay > 0, weights matches based on how recently they occurred.
        """
        if df.empty:
            return

        df = df.copy()
        if time_decay > 0:
            # Calculate days since prediction date
            df['days_diff'] = (prediction_date - df['Date']).dt.days
            df['weight'] = np.exp(-time_decay * df['days_diff'])
        else:
            df['weight'] = 1.0

        # Weighted league averages
        self.avg_home_goals = np.average(df['FTHG'], weights=df['weight'])
        self.avg_away_goals = np.average(df['FTAG'], weights=df['weight'])

        # Calculate Home Attack (weighted)
        def weighted_mean(group, col):
            return np.average(group[col], weights=group['weight'])

        home_stats = df.groupby('HomeTeam').apply(
            lambda x: pd.Series({
                'HomeAttack': weighted_mean(x, 'FTHG'),
                'HomeDefense': weighted_mean(x, 'FTAG')
            })
        )
        
        away_stats = df.groupby('AwayTeam').apply(
            lambda x: pd.Series({
                'AwayAttack': weighted_mean(x, 'FTAG'),
                'AwayDefense': weighted_mean(x, 'FTHG')
            })
        )

        # Normalize by league averages
        self.team_stats = home_stats.merge(away_stats, left_index=True, right_index=True)
        self.team_stats['HomeAttack'] /= self.avg_home_goals
        self.team_stats['AwayDefense'] /= self.avg_home_goals
        self.team_stats['AwayAttack'] /= self.avg_away_goals
        self.team_stats['HomeDefense'] /= self.avg_away_goals

    def predict_match(self, home_team, away_team):
        """
        Predicts probabilities for H/D/A and Over/Under 2.5 goals.
        """
        if self.team_stats is None or home_team not in self.team_stats.index or away_team not in self.team_stats.index:
            return None

        home_stats = self.team_stats.loc[home_team]
        away_stats = self.team_stats.loc[away_team]

        # Calculate lambda (expected goals)
        lambda_home = home_stats['HomeAttack'] * away_stats['AwayDefense'] * self.avg_home_goals
        lambda_away = away_stats['AwayAttack'] * home_stats['HomeDefense'] * self.avg_away_goals

        # Create a score matrix (up to 10 goals each)
        max_goals = 10
        score_matrix = np.outer(
            poisson.pmf(np.arange(max_goals + 1), lambda_home),
            poisson.pmf(np.arange(max_goals + 1), lambda_away)
        )

        # Probabilities
        prob_home = np.sum(np.tril(score_matrix, -1))
        prob_draw = np.sum(np.diag(score_matrix))
        prob_away = np.sum(np.triu(score_matrix, 1))

        # Over / Under 2.5
        prob_under_2_5 = 0
        for i in range(3):
            for j in range(3 - i):
                prob_under_2_5 += score_matrix[i, j]
        
        prob_over_2_5 = 1 - prob_under_2_5

        return {
            'home_prob': prob_home,
            'draw_prob': prob_draw,
            'away_prob': prob_away,
            'o25_prob': prob_over_2_5,
            'u25_prob': prob_under_2_5,
            'expected_h_goals': lambda_home,
            'expected_a_goals': lambda_away
        }

if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv('data/master_football_data.csv')
    # Train on matches except the last 10
    train_df = df.iloc[:-10]
    test_df = df.iloc[-10:]
    
    model = PoissonModel()
    last_date = test_df['Date'].max()
    model.fit(train_df, prediction_date=last_date, time_decay=0.005)
    
    print("Testing Model on last 10 matches:")
    for _, row in test_df.iterrows():
        prediction = model.predict_match(row['HomeTeam'], row['AwayTeam'])
        if prediction:
            print(f"{row['HomeTeam']} vs {row['AwayTeam']}: Predicted H {prediction['home_prob']:.2f}, Actual {row['FTR']}")
