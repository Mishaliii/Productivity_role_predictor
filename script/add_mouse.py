# script/add_mouse_intensity.py
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, "data", "train.csv")
test_path = os.path.join(BASE_DIR, "data", "test.csv")

def clean_and_add_mouse_intensity(csv_path):
    df = pd.read_csv(csv_path)

    # Fix common typos in column names
    df.columns = [c.strip().lower() for c in df.columns]
    typo_cols = [c for c in df.columns if "mouse" in c and "intens" in c]
    if typo_cols:
        print(f"⚠️ Found typo column(s) {typo_cols}, renaming to 'mouse_intensity'")
        for col in typo_cols:
            df.rename(columns={col: "mouse_intensity"}, inplace=True)

    # Add column if still missing
    if "mouse_intensity" not in df.columns:
        print(f"➕ Adding 'mouse_intensity' to {csv_path}")
        np.random.seed(42)
        df["mouse_intensity"] = (
            (df["typing_intensity"] * 0.5) +
            (df["task_switching_frequency"] * 2) +
            np.random.randint(0, 20, size=len(df))
        ).round(2)
        df["mouse_intensity"] += np.random.normal(0, 5, len(df))
        df["mouse_intensity"] = df["mouse_intensity"].clip(lower=0)

    # Save back
    df.to_csv(csv_path, index=False)
    print(f"✅ Mouse intensity fixed/added in {csv_path} ({len(df)} rows)")

# Apply to both datasets
clean_and_add_mouse_intensity(train_path)
clean_and_add_mouse_intensity(test_path)
