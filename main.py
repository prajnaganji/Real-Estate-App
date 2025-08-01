import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Import project-specific modules
from features.preprocessing import preprocess_data
from models.train_model import train_linear_model
from models.evaluate_model import evaluate_model

# Add current directory to sys.path to fix module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load CSV
file_path = "data/real_estate.csv"
if not os.path.exists(file_path):
    print("❌ File not found:", file_path)
    sys.exit()

df = pd.read_csv(file_path)
print("✅ Data loaded.")

# Preprocess data
df = preprocess_data(df)
print("✅ Data preprocessed.")

# Features and Target
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_linear_model(X_train, y_train)
print("✅ Model trained.")

# Evaluate model
mse = evaluate_model(model, X_test, y_test)
print(f"✅ Mean Squared Error: {mse}")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Starting the ML pipeline...")

# Before loading
file_path = "data/real_estate.csv"
if not os.path.exists(file_path):
    logging.error(f"File not found: {file_path}")
    sys.exit()

logging.info("Loading dataset...")
df = pd.read_csv(file_path)

try:
    logging.info("Preprocessing...")
    df = preprocess_data(df)

    X = df.drop(columns=['price'])
    y = df['price']

    logging.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logging.info("Training model...")
    model = train_linear_model(X_train, y_train)

    logging.info("Evaluating model...")
    mse = evaluate_model(model, X_test, y_test)
    logging.info(f"Mean Squared Error: {mse:.2f}")

except Exception as e:
    logging.exception("❌ Something went wrong during pipeline execution.")
    raise e

