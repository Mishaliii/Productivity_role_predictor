import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load your dataset
data = pd.read_excel("Enhanced_WorkerProductivity_Dataset.xlsx")

# Optional: Shuffle the dataset to avoid order bias
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Split into Train and Test (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Save the split datasets to the 'data/' folder
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

# (Optional) Create a sample prediction file by removing the target column
predict_sample = test_data.drop(columns=["Role"])  # Assuming 'Role' is your label
predict_sample.to_csv("data/predict_sample.csv", index=False)

print("✅ Dataset successfully split and saved.")
