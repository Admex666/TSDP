import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import RandomizedSearchCV

#%% 
def convert_to_numeric(df, predict):
    # convert non-numerical data to numerical
    le = preprocessing.LabelEncoder()
    
    df_converted = pd.DataFrame()
    x_list = []
    for col in df.columns.unique():
        if df[col].dtype == 'O':
            df_converted[col] = le.fit_transform(list(df[col]))
            x_list.append(col)
    df_num = df.copy().drop(columns=x_list)
    
    if df[predict].dtype == 'O':
        x_list.remove(predict)
        pred_col = df_converted[predict]
        df_converted.drop(columns=predict, inplace=True)
    else:
        pred_col = df_num[predict]
        df_num.drop(columns=predict, inplace=True)
    
    df_x_merged = pd.concat([df_num, df_converted], axis=1)
    
    return df_x_merged, pred_col

#%%
path = r'C:\Users\Adam\.Data files\ML py\student performance - linreg\student-mat.csv'
df = pd.read_csv(path, sep=';')
predict = 'G3'
testsize = 0.25

#%% 
x, y = convert_to_numeric(df, predict)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=testsize)

model = DecisionTreeRegressor()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

# plot_tree(model)

#%% Optimizing params
params = {'max_depth': range(2,10),
          'min_samples_split': range(2,20),
          'min_samples_leaf': range(1,15),
          'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}
rsearch = RandomizedSearchCV(estimator=model,
                             param_distributions=params,
                             n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch.best_score_)
temp = pd.DataFrame(rsearch.cv_results_)

## list(model.get_params().keys())[0]
