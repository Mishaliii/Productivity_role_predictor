import os
import pandas as pd
import pickle
import streamlit as st

# Set base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models and assets
with open(os.path.join(MODELS_DIR, "regression_model.pkl"), "rb") as f:
    reg_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "classification_model.pkl"), "rb") as f:
    clf_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "r") as f:
    feature_columns = [line.strip() for line in f.readlines()]

# Load data (used to simulate predictions)
df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Streamlit UI
def main():
    st.title("🧠 AI-Based Productivity & Role Predictor")
    st.write("Predict employee productivity score and role based on behavior metrics.")

    if st.button("🔍 Predict from Dataset"):
        X = df[feature_columns]

        # Predict
        predicted_productivity = reg_model.predict(X)
        predicted_roles = clf_model.predict(X)

        # Inverse transform roles
        role_encoder = label_encoders["Role"]
        predicted_roles = role_encoder.inverse_transform(predicted_roles)

        results = df.copy()
        results["Predicted ProductivityScore"] = predicted_productivity
        results["Predicted Role"] = predicted_roles

        st.subheader("📊 Prediction Results")
        st.dataframe(results)

        # Save output
        output_path = os.path.join(BASE_DIR, "output", "predictions.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        st.success(f"✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
