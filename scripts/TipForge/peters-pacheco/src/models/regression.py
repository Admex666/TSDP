import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class GoalRegressionModel:
    """
    SVR-based model for predicting team goals.
    Wraps two internal SVRs (Home and Away) or is instantiated twice.
    """
    
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel=kernel, C=C, epsilon=epsilon))
        ])
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the SVR model.
        """
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict goals.
        """
        return self.model.predict(X)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate performance.
        """
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return {"mae": mae, "rmse": rmse}
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)
