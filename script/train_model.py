import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --------------------------
# 1. Paths & Load dataset
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

# --------------------------
# 2. Identify target columns
# --------------------------
prod_score_col = "productivity_score"
drop_cols = [c for c in ["session_id", "productivity_label", prod_score_col, "role"] if c in df.columns]

if prod_score_col not in df.columns:
    raise ValueError(f"❌ Could not find '{prod_score_col}'. Found: {df.columns.tolist()}")

# --------------------------
# 3. Features & Targets
# --------------------------
X = df.drop(columns=drop_cols, errors="ignore")
y_reg = df[prod_score_col]

# --------------------------
# 4. Encode categorical features
# --------------------------
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# --------------------------
# 5. Scale features
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 6. Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# --------------------------
# 7. Train Regression Model
# --------------------------
reg_model = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1
)
reg_model.fit(X_train, y_train)

# Evaluate regression
y_pred = reg_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Regression Performance")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²  : {r2:.2f}")

# --------------------------
# 7b. Feature Importances
# --------------------------
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": reg_model.feature_importances_
})
feature_importances["Percentage"] = (
    feature_importances["Importance"] / feature_importances["Importance"].sum() * 100
)
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

print("\n🌟 Top 5 Important Features for Productivity Prediction:")
for rank, row in enumerate(feature_importances.head(5).itertuples(), 1):
    print(f"   {rank}. {row.Feature:<25} {row.Importance:.4f} ({row.Percentage:.2f}%)")

# Save all importances
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
imp_path = os.path.join(BASE_DIR, "output", "feature_importance.csv")
feature_importances.to_csv(imp_path, index=False)
print(f"📂 Feature importances saved at: {imp_path}")

# --------------------------
# 8. Train Clustering Model
# --------------------------
n_clusters = 4
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans_model.fit_predict(X_scaled)

# --------------------------
# 9. Map Clusters to Roles
# --------------------------
cluster_role_map = {
    0: "Developer",
    1: "Analyst",
    2: "Writer",
    3: "Designer"
}

df["cluster_id"] = clusters
df["role_name"] = df["cluster_id"].map(cluster_role_map)

print("\n📊 Clustering Results")
print(df["role_name"].value_counts())

# Save clustered dataset
clustered_path = os.path.join(BASE_DIR, "data", "train_with_clusters.csv")
df.to_csv(clustered_path, index=False)
print(f"📂 Clustered dataset saved at: {clustered_path}")

# --------------------------
# 10. Save Models & Encoders
# --------------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

joblib.dump(reg_model, os.path.join(BASE_DIR, "models", "regression_model.pkl"))
joblib.dump(kmeans_model, os.path.join(BASE_DIR, "models", "clustering_model.pkl"))
joblib.dump(label_encoders, os.path.join(BASE_DIR, "models", "label_encoders.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "models", "scaler.pkl"))
joblib.dump(cluster_role_map, os.path.join(BASE_DIR, "models", "cluster_role_map.pkl"))

# 🔹 Save feature schemas
feature_list = list(X.columns)  # realtime/logging reference
train_feature_list = list(X.columns)  # exact training schema order

joblib.dump(feature_list, os.path.join(BASE_DIR, "models", "feature_list.pkl"))
joblib.dump(train_feature_list, os.path.join(BASE_DIR, "models", "train_feature_list.pkl"))

print(f"✅ feature_list.pkl and train_feature_list.pkl saved with features: {train_feature_list}")
print("\n✅ Models, encoders, scaler, role mapping, and clustered dataset saved successfully!")
