import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
from sklearn import linear_model, preprocessing

path = r'C:\Users\Adam\.Data files\ML py\student performance - linreg\student-mat.csv'
df = pd.read_csv(path, sep=';')

#%% 
predict = 'romantic'
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

df_x_merged, y_col = convert_to_numeric(df, predict)

x = list(np.array(df_x_merged))
y = list(y_col)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)

for n in range(len(predicted)):
    print(f'Predicted: {predicted[n]}, Data: {x_test[n]}, Actual: {y_test[n]}')
print(f'Accuracy: {acc}')

#%%
df_reg = df.copy()
reg_pred_col = df_reg['G3']
df_reg.drop(columns='G3', inplace=True)

predict = 'G3'
df_x_merged, y_col = convert_to_numeric(df, predict)
x = list(np.array(df_x_merged))
y = list(y_col)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)

model = KNeighborsRegressor(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f'Accurracy: {acc}')