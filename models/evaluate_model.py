from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.base import RegressorMixin

def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates a trained regression model using Mean Squared Error.

    Parameters:
    - model: Trained regression model
    - X_test: Features to predict
    - y_test: Actual target values

    Returns:
    - Mean Squared Error
    """
    predictions = model.predict(X_test)
    return mean_squared_error(y_test, predictions)

