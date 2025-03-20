import pandas as pd
import numpy as np

sheets = ['bet_results_pred', 'bet_results_predprob', 'profits']
df_paperbets_pred = pd.read_excel('ML_PL_new/paperbets.xlsx', sheets[0])
df_paperbets_predprob = pd.read_excel('ML_PL_new/paperbets.xlsx', sheets[1])

#%% Define parameters of bankroll
m_bankroll = 1/0.03
bankroll_percent = 0.03
martingale_percent = 0.015

#%% Functions for bet sizes
def bet_size_kelly(bankroll, odds_bookie, prob_fair):
    # kelly = bankroll * ( (prob*(odds-1)) - (1-prob) / (odds-1))
    bet_size_calc = bankroll * ( (prob_fair*(odds_bookie-1)) - (1-prob_fair) ) / (odds_bookie-1)
    bet_size = 0 if bet_size_calc <= 0 else bet_size_calc
    
    return bet_size

def bet_size_fixed(balance, bankroll_percent):
    # fixed = 3% * balance
    bet_size = balance * bankroll_percent
    return bet_size

def bet_size_flat(bankroll, bankroll_percent):
    # flat = 3% * bankroll
    bet_size = bankroll * bankroll_percent
    return bet_size

def bet_size_proportional(bankroll, odds_bookie, prob_fair):
    # proportional = ( (myprob-prob) * bankroll / (odds-1))
    prob_bookie = 1/odds_bookie
    bet_size = (prob_fair - prob_bookie) * bankroll / (odds_bookie-1) if prob_fair > prob_bookie else 0
    
    return bet_size

def bet_size_martingale(bankroll, basic_percent, previous_bet, iswin):
    # martingale = 2*previous if previous == win else 1.5%
    bet_percent = previous_bet/bankroll
    if iswin:
        if bet_percent >= basic_percent*4:
            bet_size = basic_percent * bankroll
        else:
            bet_size = previous_bet * 2
    else:
        bet_size = basic_percent * bankroll
        
    return bet_size

def bet_size_my(bankroll, odds_bookie, prob_fair):
    prob_bookie = 1/odds_bookie
    bet_size = 1/(odds_bookie-1) * bankroll if prob_fair > prob_bookie else 0
    return bet_size

methods = ['kelly', 'fixed', 'flat', 'proportional', 'martingale', 'my']

#%%
cols_to_drop = [col for col in df_paperbets_predprob.columns if ('_profit' in col) or ('_bet' in col) or ('_value' in col)]
cols_not_gnb = [col for col in df_paperbets_predprob.columns if ('_prob' in col) and ('gNB' not in col)]
df_test = df_paperbets_predprob.drop(columns=cols_to_drop+cols_not_gnb).copy()

#%% Making bets 
for method in methods[:-2]:
    for out in ['H', 'D', 'A']:
        df_test[f'FTR_{out}_{method}_bet'] = None
        df_test[f'FTR_{out}_{method}_profit'] = None
        df_test[f'FTR_{out}_{method}_profit_cumsum'] = None
        for i, row in df_test.iterrows():
            odds_bookie = row[f'{out}_odds']
            prob_fair = row[f'{out}_gNB_prob']
            
            previous_bet = df_test.loc[i-1, f'FTR_{out}_{method}_bet'] if i!= 0 else 0
            #iswin = 
            #m_balance =
            
            if prob_fair >= 1/odds_bookie:
                if method == 'kelly':
                    bet_size = bet_size_kelly(m_bankroll, odds_bookie, prob_fair)
                elif method == 'fixed':
                    #bet_size = bet_size_fixed(balance, bankroll_percent)
                    bet_size = 0
                elif method == 'flat':
                    bet_size = bet_size_flat(m_bankroll, bankroll_percent)
                elif method == 'proportional':
                    bet_size = bet_size_proportional(m_bankroll, odds_bookie, prob_fair)
                elif method == 'martingale':
                    #bet_size = bet_size_martingale(m_bankroll, martingale_percent, previous_bet, iswin)
                    bet_size = 0
            else:
                bet_size = 0
            
            df_test.loc[i, f'FTR_{out}_{method}_bet'] = bet_size
            # Calc profit
            result = row['FTR_result']
            profit = bet_size * (odds_bookie-1) if result == out else - bet_size
            df_test.loc[i, f'FTR_{out}_{method}_profit'] = profit
            
            # Add to cumsum
            if i == 0:
                df_test.loc[i, f'FTR_{out}_{method}_profit_cumsum'] = profit
            else:
                df_test.loc[i, f'FTR_{out}_{method}_profit_cumsum'] = df_test.loc[i-1, f'FTR_{out}_{method}_profit_cumsum'] + profit 

for method in methods[:-2]:
    df_test[f'FTR_{method}_profits'] = 0
    for i, row in df_test.iterrows():
        df_test.loc[i, f'FTR_{method}_profits'] = np.array([df_test.loc[i, f'FTR_{out}_{method}_profit'] for out in ['H', 'D', 'A']]).sum()

        df_test[f'FTR_{method}_profits_cumsum'] = df_test[f'FTR_{method}_profits'].cumsum()
            
#%% Calculating profit/loss
