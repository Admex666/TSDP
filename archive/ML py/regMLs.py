import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

#%% define models
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

def linreg_acc(df, predict, testsize):
    x, y = convert_to_numeric(df, predict)
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=testsize)
    
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    return acc

def knnreg_acc(df, predict, testsize):
    x, y = convert_to_numeric(df, predict)
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=testsize)

    model = KNeighborsRegressor(n_neighbors=7)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    
    return acc

def dtreg_acc(df, predict, testsize):
    x, y = convert_to_numeric(df, predict)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=testsize)

    model = DecisionTreeRegressor(min_samples_split=10,
                                  min_samples_leaf=8,
                                  max_depth=6,
                                  criterion='poisson')
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    return acc

#%% Import dataframe, set parameters
path = r'C:\Users\Adam\.Data files\transfer_fee_pred\final_data.csv'
df_full = pd.read_csv(path)
df = df_full.copy().iloc[:, 4:]
df.loc[:,['current_value', 'highest_value']] = df.loc[:,['current_value', 'highest_value']]/1000000

predict = 'current_value'
testsize = 0.25

#%% Compare them
data_dict = {}
for m in ['linreg', 'knnreg', 'dtreg']:
    n = 0
    globals()[f'acc_list_{m}'] = []
    while n < 100:
        acc = globals()[f'{m}_acc'](df, predict, testsize)
        globals()[f'acc_list_{m}'].append(acc)
        n += 1
    data_dict[m] = globals()[f'acc_list_{m}']

#%% Viz
plt.title(f'Accuracy distribution of {n} samples')
plt.ylabel('Accuracy')
plt.boxplot(data_dict.values(), labels=data_dict.keys())

#%% Prediction
x, y = convert_to_numeric(df, predict)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=testsize)

model = DecisionTreeRegressor(min_samples_split=10,
                              min_samples_leaf=8,
                              max_depth=6,
                              criterion='poisson')

model.fit(x_train, y_train)

predicted = model.predict(x_test)
compare = pd.DataFrame({'predicted': predicted,
                        'real': y_test})
compare['diff'] = compare.predicted - compare.real
merged = compare.join(df_full[['team', 'name', 'position']])
