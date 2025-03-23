import os
path_base = '/'.join(os.getcwd().split('\\')[:4])
path_tsdp = path_base+'/TSDP'

os.chdir(path_tsdp)

#%%
import pandas as pd
import numpy as np

sheets = ['bet_results_pred', 'bet_results_predprob', 'profits']
df_paperbets_pred = pd.read_excel('ML_PL_new/paperbets.xlsx', sheets[0])
df_paperbets_predprob = pd.read_excel('ML_PL_new/paperbets.xlsx', sheets[1])

#%% Define parameters of bankroll
m_bankroll = 10000
bankroll_percent = 0.03
martingale_percent = 0.015

#%% Functions for bet sizes
def bet_size_kelly(bankroll, odds_bookie, prob_fair):
    # kelly = bankroll * ( (prob*(odds-1)) - (1-prob) / (odds-1))
    bet_size_calc = bankroll * ( (prob_fair*(odds_bookie-1)) - (1-prob_fair) ) / (odds_bookie-1) /2.5
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
    if iswin and bet_percent != 0:
        if bet_percent >= basic_percent*4:
            bet_size = basic_percent * bankroll
        else:
            bet_size = previous_bet * 2
    else:
        bet_size = basic_percent * bankroll
        
    return bet_size

def bet_size_my(bankroll, bankroll_percent, odds_bookie, prob_fair):
    unit = bankroll * bankroll_percent
    prob_bookie = 1/odds_bookie
    bet_size = 1/(odds_bookie-1) * unit if prob_fair > prob_bookie else 0
    return bet_size

methods = ['kelly', 'fixed', 'flat', 'proportional', 'martingale', 'my']

#%%
cols_to_drop = [col for col in df_paperbets_predprob.columns if ('_profit' in col) or ('_bet' in col) or ('_value' in col)]
cols_not_gnb = [col for col in df_paperbets_predprob.columns if ('_prob' in col) and ('gNB' not in col)]
df_test = df_paperbets_predprob.drop(columns=cols_to_drop+cols_not_gnb).copy()

#%% Making bets 
FTR_outs = ['H', 'D', 'A']
for method in methods:
    # Create columns in order
    for out in FTR_outs:
        df_test[f'FTR_{out}_{method}_bet'] = 0
    for out in FTR_outs:
        df_test[f'FTR_{out}_{method}_profit'] = 0
    df_test[f'FTR_{method}_profits'] = 0
    df_test[f'FTR_{method}_balance'] = 0
    
    for i, row in df_test.iterrows():
        for out in FTR_outs:    
            # Calc bet sizes
            odds_bookie = row[f'{out}_odds']
            prob_bookie = 1/odds_bookie
            prob_fair = row[f'{out}_gNB_prob']
            
            if i != 0:
                previous_bets = [df_test.loc[i-1, f'FTR_{col}_{method}_bet'] for col in ['H', 'D', 'A']]
                prevbets_sum = np.array(previous_bets).sum()
                count = 0
                for bet in previous_bets:
                    if bet != 0:
                        count += 1
                prevbets_number = count
                previous_bet = prevbets_sum/prevbets_number
            else:
                previous_bet = 0
            
            if i != 0:
                iswin = df_test.loc[i-1, f'FTR_{method}_profits'] >= 0
                m_balance = df_test.loc[i-1, f'FTR_{method}_balance']
            else:
                iswin = False
                m_balance = m_bankroll
            
            if prob_fair >= prob_bookie:
                if method == 'kelly':
                    bet_size = bet_size_kelly(m_bankroll, odds_bookie, prob_fair)
                elif method == 'fixed':
                    bet_size = bet_size_fixed(m_balance, bankroll_percent)
                elif method == 'flat':
                    bet_size = bet_size_flat(m_bankroll, bankroll_percent)
                elif method == 'proportional':
                    bet_size = bet_size_proportional(m_bankroll, odds_bookie, prob_fair)
                elif method == 'martingale':
                    bet_size = bet_size_martingale(m_bankroll, martingale_percent, previous_bet, iswin)
                elif method == 'my':
                    bet_size = bet_size_my(m_bankroll, bankroll_percent, odds_bookie, prob_fair)
            else:
                bet_size = 0
            
            df_test.loc[i, f'FTR_{out}_{method}_bet'] = bet_size
            
            # Calc profits for each outcome
            result = row['FTR_result']
            profit = bet_size * (odds_bookie-1) if result == out else - bet_size
            df_test.loc[i, f'FTR_{out}_{method}_profit'] = profit
       
            # Sum and cumsum profits
            if out == FTR_outs[-1]:
                profit_row = np.array([df_test.loc[i, f'FTR_{out}_{method}_profit'] for out in ['H', 'D', 'A']]).sum()
                df_test.loc[i, f'FTR_{method}_profits'] = profit_row
                
                df_test.loc[i, f'FTR_{method}_balance'] = (m_bankroll + profit_row) if i==0 else df_test.loc[i-1, f'FTR_{method}_balance'] + profit_row
                # let the bankroll be negative
                
            m_balance += profit_row
        
#%% Calculating profit/loss
import matplotlib.pyplot as plt
import seaborn as sns
balances = [column for column in df_test.columns if '_balance' in column]

sns.set_palette("husl")
plt.figure(figsize=(10,6))
plt.plot(df_test[balances], label=methods)
plt.title('Balances over time', fontsize=20)
plt.grid(axis='y')
#plt.ylim(top=23000, bottom=7000)
plt.legend()

plt.show()

profits = [column for column in df_test.columns if '_profits' in column]
profits_describe = df_test[profits].describe()
profits_describe.columns = methods

profits_describe.loc['risk_reward', :] = profits_describe.loc['std', :] / profits_describe.loc['mean', :]
