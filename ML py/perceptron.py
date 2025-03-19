from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#loading dataset 
data = load_wine()

#Splitting the dataset to train data and test data
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Making a perceptron classifier
perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron.fit(X_train, y_train)

cross_val_score(perceptron, X_train, y_train, cv=4, n_jobs=1).mean()

# Tune
perceptron.get_params()
param_grid = {
    'max_iter': [100, 500, 1000],
    'eta0': [0.001, 0.01, 0.1, 1],
    'penalty': [None, 'l1', 'l2', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01]
}

model2 = GridSearchCV(perceptron, param_grid, cv=4, n_jobs=-1, scoring='accuracy')
model2.fit(X_train, y_train)

# See results
model2.best_params_
model2.best_score_

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
plt.bar(range(len(perceptron.coef_[0])), perceptron.coef_[0])
plt.xlabel("Feature Index")
plt.ylabel("Weight Value")
plt.title("Perceptron Feature Weights")
plt.show()
