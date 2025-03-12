import pandas as pd
from sklearn import datasets
wine = datasets.load_wine(as_frame=True)

X = wine['data']
y = wine['target']

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

cross_val_score(model, X_train, y_train, cv=3, n_jobs=1).mean()

# Hypertune
model.get_params()
param_grid = {
    'n_estimators': [10,50,100,500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 9]
    }

from sklearn.model_selection import GridSearchCV
model2 = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
model2.fit(X_train, y_train)

# See results
model2.best_params_
model2.best_score_
