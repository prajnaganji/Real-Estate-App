import pandas as pd
from models.train_model import train_linear_model
from models.evaluate_model import evaluate_model

def test_linear_model():
    df = pd.DataFrame({
        'sqft': [1000, 1500, 2000],
        'baths': [1, 2, 3],
        'beds': [2, 3, 4],
        'price': [200000, 300000, 400000]
    })

    X = df[['sqft', 'baths', 'beds']]
    y = df['price']

    model = train_linear_model(X, y)
    mse = evaluate_model(model, X, y)

    assert mse < 1e-3, f"Unexpectedly high MSE: {mse}"
