from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Trains a Linear Regression model.

    Parameters:
    - X_train: DataFrame of input features
    - y_train: Series of target variable

    Returns:
    - Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

