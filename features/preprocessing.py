import pandas as pd

def preprocess_data(df):
    # Drop missing values
    df = df.dropna()

    # Remove invalid rows
    df = df[df['price'] > 0]
    df = df[df['sqft'] > 0]

    # One-hot encode categorical columns (only if they exist)
    if 'property_type' in df.columns:
        df = pd.get_dummies(df, columns=['property_type'], drop_first=True)

    # Reset index to avoid misalignment issues
    df = df.reset_index(drop=True)

    return df
