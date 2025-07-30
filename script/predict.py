import pandas as pd
import joblib
import os

# ----------- File Paths -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'test.csv')
REG_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'regression_model.pkl')
CLS_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'classification_model.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'output', 'predictions.csv')

# ----------- Load Test Data -----------
test_df = pd.read_csv(DATA_PATH)

# Backup Employee_ID if needed later
if 'Employee_ID' in test_df.columns:
    employee_ids = test_df['Employee_ID']
    test_df = test_df.drop('Employee_ID', axis=1)
else:
    employee_ids = pd.Series(range(len(test_df)))

# Drop actual labels if accidentally included
for col in ['Productivity_Score', 'Role']:
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)

# ----------- Encode Categorical Columns -----------
cat_cols = test_df.select_dtypes(include='object').columns
for col in cat_cols:
    test_df[col] = test_df[col].astype('category').cat.codes

# ----------- Load Trained Models -----------
try:
    reg_model = joblib.load(REG_MODEL_PATH)
    cls_model = joblib.load(CLS_MODEL_PATH)
except FileNotFoundError as e:
    print("❌ Model file not found:", e)
    exit()

# ----------- Make Predictions -----------
try:
    predicted_productivity = reg_model.predict(test_df)
    predicted_role = cls_model.predict(test_df)
except Exception as e:
    print("❌ Error during prediction:", e)
    exit()

# ----------- Prepare Results -----------
results_df = pd.DataFrame({
    'Employee_ID': employee_ids,
    'Predicted_Productivity_Score': predicted_productivity,
    'Predicted_Role': predicted_role
})

# ----------- Save & Display Results -----------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Predictions Completed Successfully!")
print(results_df.head())
print(f"\n📁 Saved to: {OUTPUT_PATH}")
