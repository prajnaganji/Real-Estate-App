import pandas as pd
def preprocess_data(df):
    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['sqft'] > 0]

    # One-hot encode categorical columns (e.g., property_type)
    df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

    return df
