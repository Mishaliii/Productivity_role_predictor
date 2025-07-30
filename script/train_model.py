import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("data/Enhanced_WorkerProductivity_Dataset.xlsx")

# Clean column names
df.rename(columns=lambda x: x.strip(), inplace=True)
df.columns = df.columns.str.replace(" ", "")

# Label encoding for Role
role_encoder = LabelEncoder()
df["RoleEncoded"] = role_encoder.fit_transform(df["Role"])

# Create ProductivityLevel from ProductivityScore
def map_productivity(score):
    if score < 50:
        return "Bad"
    elif score < 70:
        return "Neutral"
    else:
        return "Good"

df["ProductivityLevel"] = df["ProductivityScore"].apply(map_productivity)
prod_encoder = LabelEncoder()
df["ProductivityLevelEncoded"] = prod_encoder.fit_transform(df["ProductivityLevel"])

# Save encoders
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump({
        "role_encoder": role_encoder,
        "prod_encoder": prod_encoder
    }, f)

# Features and targets
X = df.drop(["ProductivityScore", "Role", "RoleEncoded", "ProductivityLevel", "ProductivityLevelEncoded"], axis=1)
y_score = df["ProductivityScore"]
y_role = df["RoleEncoded"]
y_prod_level = df["ProductivityLevelEncoded"]

# Save feature columns for use in app
with open("models/feature_columns.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

# Train regression model (ProductivityScore)
reg_model = RandomForestRegressor()
reg_model.fit(X, y_score)
with open("models/regression_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)

# Train role classification model
clf_role = RandomForestClassifier()
clf_role.fit(X, y_role)
with open("models/classification_role_model.pkl", "wb") as f:
    pickle.dump(clf_role, f)

# Train productivity level classification model
clf_prod = RandomForestClassifier()
clf_prod.fit(X, y_prod_level)
with open("models/classification_productivity_model.pkl", "wb") as f:
    pickle.dump(clf_prod, f)

print("✅ All models trained and saved: Regression, Role Classifier, Productivity Level Classifier")
