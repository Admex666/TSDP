import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
wd_old = os.getcwd()
if wd_old != 'C:\\Users\\Adam\\..Data\\TSDP':
    wd_base = wd_old.split('\\')[:4]
    wd_new = '\\'.join(wd_base)+'\\TSDP'
    os.chdir(wd_new)
from fbref import fbref_module as fbref
from ML_PL_new.ML_PL_transform_data import df_to_model_input 

#%% Loading data from website
seasons = ['2223', '2324']
leagues = ['E0', 'SP1', 'I1', 'D1']

df = pd.DataFrame()
for league in leagues:
    for season in seasons:
        url_table = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        table = pd.read_csv(url_table)
        table['Season'] = season
        df = pd.concat([df, table], ignore_index=True)
    
    
# Only needed columns
needed_cols = ['Date', 'Div', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
betting_cols = ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5']
df = df[needed_cols+betting_cols]
# create BTTS and O2,5 labels
df['BTTS'] = np.where((df.FTHG!=0)&(df.FTAG!=0),'Yes','No')
df['O/U2.5'] = np.where(df.FTHG+df.FTAG>2.5,'Over','Under')

#%% Transforming data
model_input = pd.DataFrame()
for league in leagues:
    for season in seasons:
        df_part = df[(df.Div == league) & (df.Season == season)]
        model_input_part = df_to_model_input(df_part, weather=False)
        model_input = pd.concat([model_input, model_input_part], ignore_index=True)
#model_input.to_csv('ML_PL_new/df_tr.csv', index=False)

# Prepare ML model
params_grid = {'GaussianNB':{},
               'DecisionTreeClassifier': {'max_depth': 4, 'min_samples_split': 6,
                                          'min_samples_leaf': 2, 'criterion':'entropy'
                                          },
               'RandomForestClassifier': {'max_depth': 7, 'min_samples_split': 5,
                                          'min_samples_leaf': 2, 'max_features': 'sqrt', 
                                          'bootstrap': True      
                                          },
               'KNeighborsClassifier': {'algorithm': 'brute',
                                        'leaf_size': 10,
                                        'n_neighbors': 8,
                                        'p': 2,
                                        'weights': 'distance'
                                        },
               'GradientBoostingClassifier': {'n_estimators': 500,
                                              'learning_rate': 0.001,
                                              'max_depth': 3
                                               },
               'Perceptron': {'alpha': 0.01, 'eta0': 0.001,
                              'max_iter': 100, 'penalty': 'l1'}
               }

model_short_dict = {'GaussianNB':'gNB', 
                    'RandomForestClassifier': 'RF',
                    'DecisionTreeClassifier': 'DT',
                    'KNeighborsClassifier': 'KNN',
                    #'GradientBoostingClassifier': 'GB',
                    #'Perceptron': 'ptron'
                    }

model_list = list(model_short_dict.keys())

#%% Build and test ML model
df_accs = pd.DataFrame()

for btype in ['FTR', 'BTTS', 'O/U2.5']:
    x = model_input.iloc[:,6:]
    y = model_input.loc[:, btype]
    for m in model_list:
        acc_list = []
        m_short = model_short_dict[m]
        for n in range(600, 800):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=n)
            
            model = globals()[f"{m}"]()
            model.set_params(**params_grid[m])
            
            model.fit(x_train, y_train)
            if n % 20 == 0:
                print(f'{btype}, {m}, {n}')
            
# Evaluation
            y_pred = model.predict(x_test)
            accuracy= accuracy_score(y_pred, y_test)
            acc_list.append(accuracy)
        df_accs[f'{btype}_{m_short}'] = acc_list
    
accs_describe = df_accs.describe().iloc[1:, :]
df_accs.boxplot()

accs_describe_relative = pd.DataFrame()
for btype in ['FTR', 'BTTS', 'O/U2.5']:
    for m in model_short_dict.values():
        if btype == 'FTR':
            denominator = 1/3
        else:
            denominator = 1/2
        accs_describe_relative[f'{btype}_{m}'] = round((accs_describe[f'{btype}_{m}'] / denominator -1)*100,1)

#%% Difference between accuracies
output_path = 'ML_PL_new/test_scores.xlsx'
accs_describe_prev = pd.read_excel(output_path, index_col=0)

accs_diff = pd.DataFrame()
for col in accs_describe:
    for i in accs_describe.index:
        if (i in accs_describe_prev.index) & (col in accs_describe_prev.columns):
            val_prev = accs_describe_prev.loc[i, col]
            val_new = accs_describe.loc[i, col]
            diff = val_new / val_prev  - 1 
            accs_diff.loc[i, col] = diff

#%% Test profit/loss
df.rename(columns={'B365H': 'H_odds',
                    'B365D': 'D_odds',
                    'B365A': 'A_odds',
                    'B365>2.5': 'Over_odds',
                    'B365<2.5': 'Under_odds'
                    },
               inplace=True)

def calculate_profit(y_true, y_pred, df, bet_type):
    profit = 0
    for (idx, actual), pred in zip(y_true.items(), y_pred):
        row = df.loc[idx]
        if bet_type == 'FTR':
            odds = row[['H_odds', 'D_odds', 'A_odds']][['H', 'D', 'A'].index(pred)]
        elif bet_type == 'O/U2.5':
            odds = row['Over_odds'] if pred == 'Over' else row['Under_odds']
        
        stake = 1 / (odds - 1) if odds > 1 else 0
        if pred == actual:
            profit += (odds - 1) * stake  
        else:
            profit -= stake
    
    return profit


df_profits = pd.DataFrame()

for btype in ['FTR', 'O/U2.5']:
    x = model_input.iloc[:,6:-5]  # Excluding odds columns
    y = model_input.loc[:, btype]
    odds_data = df[['H_odds', 'D_odds', 'A_odds', 'Over_odds', 'Under_odds']]
    
    for m in model_list:
        profit_list = []
        m_short = model_short_dict[m]
        for n in range(1000, 1100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=n)
            odds_test = odds_data.loc[y_test.index]
            
            model = globals()[f"{m}"]()
            model.set_params(**params_grid[m])
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            profit = calculate_profit(y_test, y_pred, df, btype)
            profit_list.append(profit)
            
        df_profits[f'{btype}_{m_short}'] = profit_list
profits_describe = df_profits.describe()

#%% Saving to excel
with pd.ExcelWriter(output_path) as writer:
    accs_describe.to_excel(writer, sheet_name='describe')
    accs_describe_relative.to_excel(writer, sheet_name='describe_relative')
    profits_describe.to_excel(writer, sheet_name='profit_stats')

#%% Tune hyperparameters
"""
params_all = list(model.get_params().keys())
params = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [2, 5, 10, 20],
    'n_neighbors': [3, 5, 6, 7, 8, 9, 10, 11, 13, 15],
    'p': [1, 2, 3, 4],
    'weights': ['uniform', 'distance']
}
rsearch = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch.best_score_)
rs_results = pd.DataFrame(rsearch.cv_results_)

#%% Tune GB
param_grid = {
    'n_estimators': [10,50,100,500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 9]
    }

model2 = GridSearchCV(model, param_grid, cv=4, n_jobs=-1)
model2.fit(x_train, y_train)

# See results
print('Best params:',model2.best_params_, '\nScore:', model2.best_score_)
"""