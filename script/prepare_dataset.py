import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Load Excel dataset
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "Enhanced_WorkerProductivity_Dataset.xlsx")

df = pd.read_excel(EXCEL_PATH)

print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# --------------------------
# 2. Normalize column names
# --------------------------
df.columns = [col.strip().lower() for col in df.columns]

# --------------------------
# 3. Train-test split (80/20)
# --------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --------------------------
# 4. Save train and test CSVs
# --------------------------
train_path = os.path.join(BASE_DIR, "data", "train.csv")
test_path = os.path.join(BASE_DIR, "data", "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ train.csv saved: {train_df.shape}")
print(f"✅ test.csv saved: {test_df.shape}")
