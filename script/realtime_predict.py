# script/realtime_predict.py

import pandas as pd
import joblib
import os
from datetime import datetime
from script.collect_realtime_data import collect_realtime_data

# Paths
CLASSIFICATION_MODEL_PATH = "models/classification_model.pkl"
REGRESSION_MODEL_PATH = "models/regression_model.pkl"
ENCODER_PATH = "models/encoder.pkl"   # if used
FEATURE_LIST_PATH = "models/feature_list.pkl"
OUTPUT_PATH = "output/realtime_predictions.csv"

# Load models
classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
regression_model = joblib.load(REGRESSION_MODEL_PATH)

# Load encoder (optional, only if categorical features were encoded)
if os.path.exists(ENCODER_PATH):
    encoder = joblib.load(ENCODER_PATH)
else:
    encoder = None

# Load feature list
feature_list = joblib.load(FEATURE_LIST_PATH)

# Collect real-time data
realtime_df = collect_realtime_data()

# Preprocess real-time data to match training features
X_real = realtime_df.copy()
X_real = X_real[feature_list]  # align columns

# If encoder was used (e.g., for categorical columns), apply transformation
if encoder:
    X_real = encoder.transform(X_real)

# Predict
predicted_role = classification_model.predict(X_real)[0]
predicted_productivity = regression_model.predict(X_real)[0]

# Add predictions to DataFrame
realtime_df["PredictedRole"] = predicted_role
realtime_df["PredictedProductivityScore"] = round(predicted_productivity, 2)

# Save results to CSV (append mode)
if not os.path.exists(OUTPUT_PATH):
    realtime_df.to_csv(OUTPUT_PATH, index=False)
else:
    realtime_df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)

# Print result to console (optional)
print("\nReal-Time Prediction:")
print(realtime_df[["Timestamp", "PredictedRole", "PredictedProductivityScore"]])
