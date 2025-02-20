import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Adam\.Data files\student performance - linreg\student-mat.csv',sep=';')
df = df.loc[:,['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3' # G3 is final grade

#%% Create linreg
def linreg_acc(df, predict):
    x = df.drop(columns=predict)
    y = df[predict]
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.25)
    
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    return(acc)
    
# predictions = linear.predict(x_test)

#%% Run it
n = 0
acc_list = []
while n < 200:
    acc_list.append(linreg_acc(df))
    n += 1

plt.title(f'Distribution of accuracy of {n} samples')
plt.xlabel('Accuracy')
plt.hist(acc_list)
