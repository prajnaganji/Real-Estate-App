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
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write(" Raw Input Data", df.head())

    try:
        # Preprocess the data
        df_processed = preprocess_data(df)
        X = df_processed.drop("price", axis=1)
        y = df_processed["price"]

        # Train model
        model = train_linear_model(X, y)

        # Predict prices
        predictions = model.predict(X)
        df_processed['Predicted_Price'] = predictions

        # Display predictions
        st.success(" Prediction Successful")
        st.write(" Predicted Results", df_processed[['Predicted_Price']].head())

    except Exception as e:
        st.error(f" Error: {e}")

