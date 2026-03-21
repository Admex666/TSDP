import pandas as pd
import numpy as np

def main():
    df = pd.read_csv(r"E:\Data\TSDP\scripts\TipForge\NBA\reddit_0321\full_backtest_results.csv")
    
    # Filter to placed bets only
    bets = df[df['bet_placed'] != 'No Bet'].copy()
    
    # Calculate necessary variables
    bets['p'] = np.where(bets['bet_placed'] == 'Home', bets['pred_prob_home'], 1 - bets['pred_prob_home'])
    bets['odds'] = np.where(bets['bet_placed'] == 'Home', bets['odds_home'], bets['odds_away'])
    bets['implied'] = 1 / bets['odds']
    bets['edge'] = bets['p'] - bets['implied']
    bets['b'] = bets['odds'] - 1
    bets['q'] = 1 - bets['p']
    
    # Kelly Formula: f = (p * b - q) / b
    bets['kelly_fraction'] = (bets['p'] * bets['b'] - bets['q']) / bets['b']
    
    # Assume 1% of bankroll = 1 Unit. So Kelly fraction * 100 = Bet Units
    bets['kelly_units'] = bets['kelly_fraction'] * 100
    
    # Strategies Array
    strats = []
    
    # 1. Flat Betting (1 Unit)
    bets['flat_risk'] = 1.0
    bets['flat_profit'] = np.where(bets['won_ML'] == 1, bets['flat_risk'] * bets['b'], -bets['flat_risk'])
    
    strats.append({
        'Strategy': 'Flat (1U)',
        'Total Bets': len(bets),
        'Units Risked': bets['flat_risk'].sum(),
        'Profit (U)': bets['flat_profit'].sum(),
        'ROI': (bets['flat_profit'].sum() / bets['flat_risk'].sum()) * 100
    })
    
    # 2. Full Kelly (Units Risked = kelly_fraction * 100)
    bets['kelly_risk'] = bets['kelly_units']
    bets['kelly_profit'] = np.where(bets['won_ML'] == 1, bets['kelly_risk'] * bets['b'], -bets['kelly_risk'])
    
    strats.append({
        'Strategy': 'Full Kelly',
        'Total Bets': len(bets),
        'Units Risked': bets['kelly_risk'].sum(),
        'Profit (U)': bets['kelly_profit'].sum(),
        'ROI': (bets['kelly_profit'].sum() / bets['kelly_risk'].sum()) * 100
    })
    
    # 3. 1/4 Kelly
    bets['quarter_kelly_risk'] = bets['kelly_units'] / 4
    bets['quarter_kelly_profit'] = np.where(bets['won_ML'] == 1, bets['quarter_kelly_risk'] * bets['b'], -bets['quarter_kelly_risk'])
    
    strats.append({
        'Strategy': '1/4 Kelly',
        'Total Bets': len(bets),
        'Units Risked': bets['quarter_kelly_risk'].sum(),
        'Profit (U)': bets['quarter_kelly_profit'].sum(),
        'ROI': (bets['quarter_kelly_profit'].sum() / bets['quarter_kelly_risk'].sum()) * 100
    })
    
    # 4. Confidence Tiers
    # Less than 4% edge = 1U, 4-7% edge = 2U, >7% = 3U
    conditions = [
        (bets['edge'] > 0.07),
        (bets['edge'] >= 0.04) & (bets['edge'] <= 0.07),
        (bets['edge'] < 0.04)
    ]
    choices = [3.0, 2.0, 1.0]
    bets['tier_risk'] = np.select(conditions, choices, default=1.0)
    bets['tier_profit'] = np.where(bets['won_ML'] == 1, bets['tier_risk'] * bets['b'], -bets['tier_risk'])
    
    strats.append({
        'Strategy': 'Confidence Tiers (1-3U)',
        'Total Bets': len(bets),
        'Units Risked': bets['tier_risk'].sum(),
        'Profit (U)': bets['tier_profit'].sum(),
        'ROI': (bets['tier_profit'].sum() / bets['tier_risk'].sum()) * 100
    })
    
    summary = pd.DataFrame(strats)
    print("\n--- Bet Sizing Strategy Comparison ---")
    print(summary.to_string(index=False, formatters={'Units Risked': '{:.1f}'.format, 'Profit (U)': '{:.2f}'.format, 'ROI': '{:.2f}%'.format}))
    
if __name__ == "__main__":
    main()
