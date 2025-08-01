import streamlit as st
import pandas as pd
import numpy as np
import joblib
from features.preprocessing import preprocess_data
from models.train_model import train_linear_model

# Page title
st.title(" Real Estate Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your real estate CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Raw Input Data", df.head())

    try:
        # Preprocess
        df_processed = preprocess_data(df)

        # Drop rows with missing or invalid values if necessary
        df_processed = df_processed.dropna()

        X = df_processed.drop("price", axis=1)
        y = df_processed["price"]

        # Train model
        model = train_linear_model(X, y)

        # Predict
        prediction = model.predict(X)

        # Match rows between processed and original
        df_result = df.loc[df_processed.index].copy()
        df_result['Predicted_Price'] = prediction

        st.success(" Prediction Successful")
        st.write(" Predicted Results", df_result[['Predicted_Price']].head())

    except Exception as e:
        st.error(f" Error: {e}")
