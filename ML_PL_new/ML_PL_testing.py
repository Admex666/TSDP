import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from fbref import fbref_module as fbref
from ML_PL_new import ML_PL_transform_data as mlpl 

# Loading data from website
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)
# Only needed columns
needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'FTR']
df = df[needed_cols]
# create BTTS and O2,5 labels
df['BTTS'] = np.where((df.FTHG!=0)&(df.FTAG!=0),'Yes','No')
df['O/U2.5'] = np.where(df.FTHG+df.FTAG>2.5,'Over','Under')

#%% Transforming data
model_input = mlpl.df_to_model_input(df)

# Prepare ML model
df_accs = pd.DataFrame()

params_grid = {'GaussianNB':{},
               'DecisionTreeClassifier': {'max_depth': 4, 'min_samples_split': 6,
                                          'min_samples_leaf': 2, 'criterion':'entropy'
                                          },
               'RandomForestClassifier': {'max_depth': 7, 'min_samples_split': 5,
                                          'min_samples_leaf': 2, 'max_features': 'sqrt', 
                                          'bootstrap': True      
                                          }
               }
model_short_dict = {'GaussianNB':'gNB', 
                    'RandomForestClassifier': 'RF',
                    'DecisionTreeClassifier': 'DT'}

#%% Build and test ML model
for btype in ['FTR', 'BTTS', 'O/U2.5']:
    x = model_input.iloc[:,6:]
    y = model_input.loc[:, btype]
    for m in ['GaussianNB', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        acc_list = []
        m_short = model_short_dict[m]
        for n in range(170, 220):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=n)
            
            model = globals()[f"{m}"]()
            model.set_params(**params_grid[m])
            
            model.fit(x_train, y_train)
            
# Evaluation
            y_pred = model.predict(x_test)
            accuracy= accuracy_score(y_pred, y_test)
            acc_list.append(accuracy)
        df_accs[f'{btype}_{m_short}'] = acc_list
    
accs_describe = df_accs.describe().iloc[1:, :]
df_accs.boxplot(rot=90)
    
#%% Optimizing parameters
params_all = list(model.get_params().keys())
params = {
    'max_depth': [None, 5, 7, 10, 12, 15, 17, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
rsearch = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch.best_score_)
rs_results = pd.DataFrame(rsearch.cv_results_)
