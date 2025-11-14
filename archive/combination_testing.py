import pandas as pd
import itertools
import numpy as np
from datetime import datetime, date
import random

realbets = pd.read_excel(r'C:\Users\Adam\Dropbox\TSDP_output\.paperbetting.xlsx', sheet_name='VALÓS v2')
realbets.dropna(subset='Eredmény', inplace=True)

#%% Kombók létrehozása
# Napi meccsek
rb_daily = realbets[realbets['Dátum'].dt.date == datetime(2025, 4, 20).date()]
rb_daily_nrs = rb_daily['Nr. #'].copy().to_list()
# Alapvető P/L kiszámítása
profit_normal = rb_daily['P/L'].sum()
tet_normal = rb_daily['Tét'].sum()
print(f'A normál profit: {profit_normal:.0f}')

profitok = []
stats_win = {}
stats_lose = {}
# Többféle kombináció tesztelése
for seed in range(0, 300):
    random.seed(seed)
    random.shuffle(rb_daily_nrs)
    
    # Példa szorzók (10 mérkőzésre)
    szelvenyek_nr = [rb_daily_nrs[i:i+4] for i in range(0, len(rb_daily_nrs) - len(rb_daily_nrs) % 4, 4)]
    szorzok = [rb_daily.loc[rb_daily['Nr. #'] == nr, 'Szorzó'].iloc[0] for nr in rb_daily_nrs]
    szelvenyek = [szorzok[i:i+4] for i in range(0, len(szorzok) - len(szorzok) % 4, 4)]
    outcomeok = [rb_daily.loc[rb_daily['Nr. #'] == nr, 'Eredmény'].iloc[0] for nr in rb_daily_nrs]
    szelvenyek_out = [outcomeok[i:i+4] for i in range(0, len(outcomeok) - len(outcomeok) % 4, 4)]
    
    # Kombinációk generálása (3/4-es kombinációk)
    szelveny_kombik = {}
    for i, szelveny, szelveny_out in zip(range(len(szelvenyek)), szelvenyek, szelvenyek_out):
        kombik = list(itertools.combinations(szelveny, 3))
        kombik_out = list(itertools.combinations(szelveny_out, 3))
        szelveny_kombik[i] = {'kombik':kombik, 'kombik_out': kombik_out,
                              'szorzó': [np.prod(kombik[n]) for n in range(len(kombik))],
                              'szorzó_stat': {'min': min(szelveny),
                                              'std': pd.Series(szelveny).std(),
                                              'mean': pd.Series(szelveny).mean(),
                                              'max': max(szelveny)},
                              }
    
    # Eredmények kiszámítása
    tet = tet_normal / (len(rb_daily_nrs) - len(rb_daily_nrs) % 4)
    tet_ossz = 0
    profit_napi = 0
    for szelo_nr in szelveny_kombik.keys():
        profit_szelveny = 0
        for n in range(4):
            tet_ossz += tet
            win = all(x=='W' for x in szelveny_kombik[szelo_nr]['kombik_out'][n])
            if win:
                szorzo = szelveny_kombik[szelo_nr]['szorzó'][n]
                profit = tet*szorzo - tet
                stats_win[len(stats_win)] = szelveny_kombik[szelo_nr]['szorzó_stat']
            else:
                profit = - tet
                stats_lose[len(stats_lose)] = szelveny_kombik[szelo_nr]['szorzó_stat']
            profit_szelveny += profit
        profit_napi += profit_szelveny
    #print(f'Napi profit: {profit_napi:.0f}')
    #print(f'Összes tét: {tet_ossz}')
    profitok.append(profit_napi)

# Leíró statisztika
profitok_pd = pd.Series(profitok)
profitok_desc = profitok_pd.describe()
profitok_desc_rel = profit_normal / profitok_desc.iloc[1:]
# Percentilis kiszámítása
from scipy.stats import percentileofscore
print(f'{int(percentileofscore(profitok_pd, profit_normal))}. percentilisben van a normál profit.')

#%% Stats win és stats lose elemzése
df_stats_win = np.transpose(pd.DataFrame(stats_win))
df_stats_lose = np.transpose(pd.DataFrame(stats_lose))

df_stats_win.describe()
df_stats_lose.describe()

#%% Plot
import matplotlib.pyplot as plt
plt.boxplot(profitok)
plt.show()

#%% Random csoportosítás helyett legkisebb szórás
from itertools import permutations
import numpy as np

def min_std_grouping(nums):
    min_std = float('inf')
    best_groups = None
    
    for perm in set(permutations(nums)):
        group1 = perm[0:4]
        group2 = perm[4:8]
        group3 = perm[8:12]

        means = [np.mean(group1), np.mean(group2), np.mean(group3)]
        std = np.std(means)

        if std < min_std:
            min_std = std
            best_groups = [group1, group2, group3]

            if std == 0:
                break  # nem lehet jobb

    return best_groups, [np.mean(g) for g in best_groups], min_std

# Példa
nr_odds_dict = dict(zip(rb_daily['Nr. #'], rb_daily['Szorzó']))
nums = [1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4]
groups, avgs, std = min_std_grouping(nums)

print("Legjobb csoportok:", groups)
print("Csoportok átlaga:", avgs)
print("Átlagok szórása:", std)
