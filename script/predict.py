import pandas as pd
import joblib
import os

# ----------- File Paths -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'test.csv')
REG_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'regression_model.pkl')
CLS_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'classification_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, '..', 'models', 'label_encoders.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'feature_list.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'output', 'predictions.csv')

# ----------- Load Models & Encoders -----------
reg_model = joblib.load(REG_MODEL_PATH)
cls_model = joblib.load(CLS_MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
feature_list = joblib.load(FEATURES_PATH)

# ----------- Helper Function for Prediction -----------
def prepare_data(df):
    """Apply label encoding and align features to training data."""
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = le.transform(df[col])
    # Add missing columns
    missing_cols = set(feature_list) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    return df[feature_list]

def predict_batch():
    """Batch prediction from test.csv."""
    test_df = pd.read_csv(DATA_PATH)

    # Backup Employee_ID
    if 'Employee_ID' in test_df.columns:
        employee_ids = test_df['Employee_ID']
        test_df.drop('Employee_ID', axis=1, inplace=True)
    else:
        employee_ids = pd.Series(range(len(test_df)))

    # Remove target columns if exist
    for col in ['Productivity_Score', 'Role']:
        if col in test_df.columns:
            test_df.drop(col, axis=1, inplace=True)

    # Prepare and predict
    test_df = prepare_data(test_df)
    predicted_productivity = reg_model.predict(test_df)
    predicted_role = cls_model.predict(test_df)

    results_df = pd.DataFrame({
        'Employee_ID': employee_ids,
        'Predicted_Productivity_Score': predicted_productivity,
        'Predicted_Role': predicted_role
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\n✅ Batch Predictions Completed Successfully!")
    print(results_df.head())
    print(f"\n📁 Saved to: {OUTPUT_PATH}")

def predict_realtime(input_dict):
    """Real-time prediction from a single row dictionary."""
    df = pd.DataFrame([input_dict])
    df = prepare_data(df)

    productivity = reg_model.predict(df)[0]
    role = cls_model.predict(df)[0]

    print("\n⚡ Real-Time Prediction:")
    print(f"Predicted Productivity Score: {productivity:.2f}")
    print(f"Predicted Role: {role}")
    return productivity, role

# ----------- Choose Mode -----------
if __name__ == "__main__":
    mode = input("Enter mode (batch/realtime): ").strip().lower()

    if mode == "batch":
        predict_batch()
    elif mode == "realtime":
        from collect_realtime_data import get_employee_data
        input_data = get_employee_data()  # Get simulated real-time data
        predict_realtime(input_data)
    else:
        print("❌ Invalid mode. Use 'batch' or 'realtime'.")
