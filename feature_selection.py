def preprocess_data(df):
    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['sqft'] > 0]
    return df
