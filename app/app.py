import os
import pandas as pd
import streamlit as st
import joblib
import sys
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from script.collect_realtime_data import get_realtime_data

# ================= CONFIG =================
PRODUCTIVITY_THRESHOLD = 70
SESSION_ID = int(time.time())  # unique session per run
LOG_FILE = "output/realtime_log.csv"

# ================= LOAD MODELS =================
try:
    reg_model = joblib.load("models/regression_model.pkl")
    cluster_model = joblib.load("models/clustering_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_list = joblib.load("models/feature_list.pkl")             # realtime schema
    train_feature_list = joblib.load("models/train_feature_list.pkl") # training schema
    cluster_role_map = joblib.load("models/cluster_role_map.pkl")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ================= STREAMLIT CONFIG =================
st.set_page_config(page_title="⚡ Productivity & Role Clustering", layout="wide")
st.title("⚡ AI-Based Productivity & Role Clustering System")

# Sidebar mode selection
mode = st.sidebar.radio(
    "Choose Mode",
    ["🔴 Real-Time Prediction", "👥 Manager View (All Devices)", "📂 Test Dataset Prediction", "📊 Feature Importance"]
)

# ================= REALTIME MODE =================
if mode == "🔴 Real-Time Prediction":
    st_autorefresh(interval=5000, key="realtime_refresh")
    st.subheader("🔴 Live Device Prediction")

    try:
        latest_row = get_realtime_data()

        # Debug capture
        st.subheader("🧩 Raw Feature Capture (Debug)")
        st.write(latest_row)

        # Ensure training features exist
        for col in train_feature_list:
            if col not in latest_row.columns:
                latest_row[col] = 0

        # Align schema
        X = latest_row[train_feature_list].astype(float)
        X_scaled = scaler.transform(X)

        # Predictions
        predicted_productivity = round(float(reg_model.predict(X_scaled)[0]), 2)
        predicted_cluster = int(cluster_model.predict(X_scaled)[0])
        predicted_role = cluster_role_map.get(predicted_cluster, f"Cluster {predicted_cluster}")

        # Role grouping
        role_group_map = {
            "Developer": "Technical",
            "Engineer": "Technical",
            "Designer": "Creative",
            "Writer": "Creative",
            "Manager": "Business",
            "Analyst": "Business"
        }
        role_group = role_group_map.get(predicted_role, predicted_role)

        status = "✅ Productive" if predicted_productivity >= PRODUCTIVITY_THRESHOLD else "❌ Not Productive"

        # Compose row
        save_row = latest_row.copy().reset_index(drop=True)
        save_row["PredictedProductivity"] = predicted_productivity
        save_row["ClusterID"] = predicted_cluster
        save_row["RoleName"] = predicted_role
        save_row["RoleGroup"] = role_group
        save_row["ProductivityStatus"] = status
        save_row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_row["SessionID"] = SESSION_ID

        # Save log
        os.makedirs("output", exist_ok=True)
        if not os.path.exists(LOG_FILE):
            save_row.to_csv(LOG_FILE, index=False)
        else:
            save_row.to_csv(LOG_FILE, mode="a", header=False, index=False)

        # Load logs (this session only, this device)
        try:
            df_log = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            if "Timestamp" in df_log.columns:
                df_log["Timestamp"] = pd.to_datetime(df_log["Timestamp"], errors="coerce")
                df_log = df_log.dropna(subset=["Timestamp"])
            df_log = df_log[df_log["SessionID"] == SESSION_ID]  # filter session
            df_log = df_log.sort_values("Timestamp", ascending=False)
        except Exception as e:
            st.error(f"⚠️ Could not load log file: {e}")
            df_log = save_row.copy()

        # ================= Metrics =================
        session_avg = df_log["PredictedProductivity"].mean() if "PredictedProductivity" in df_log else predicted_productivity
        total_switches = df_log["task_switch_delta"].sum() if "task_switch_delta" in df_log else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Latest Productivity", f"{predicted_productivity:.2f}")
        col2.metric("Session Avg Productivity", f"{session_avg:.2f}")
        col3.metric("Role", predicted_role)
        col4.metric("Status", status)
        col5.metric("Total Switches", f"{int(total_switches)}")

        # ================= Logs =================
        st.subheader("📌 Latest Logs (This Session Only)")
        st.dataframe(df_log.head(15).reset_index(drop=True), use_container_width=True)

        # ================= Charts =================
        if "PredictedProductivity" in df_log.columns:
            st.subheader("📈 Productivity Trend")
            st.line_chart(df_log.set_index("Timestamp").sort_index()["PredictedProductivity"].astype(float))

        if "task_switch_total" in df_log.columns:
            st.subheader("🔄 Task Switching (Cumulative)")
            st.line_chart(df_log.set_index("Timestamp").sort_index()["task_switch_total"].astype(int))

        if "idle_duration" in df_log.columns:
            st.subheader("⏳ Idle Duration Trend (sec)")
            st.line_chart(df_log.set_index("Timestamp").sort_index()["idle_duration"].astype(float))

    except Exception as e:
        st.error(f"❌ Error in real-time prediction: {e}")

# ================= MANAGER MODE =================
elif mode == "👥 Manager View (All Devices)":
    st.subheader("👥 Manager View — All Devices")

    try:
        if not os.path.exists(LOG_FILE):
            st.warning("⚠️ No logs found yet. Start real-time mode on some devices.")
        else:
            df_log = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            if "Timestamp" in df_log.columns:
                df_log["Timestamp"] = pd.to_datetime(df_log["Timestamp"], errors="coerce")
                df_log = df_log.dropna(subset=["Timestamp"]).sort_values("Timestamp", ascending=False)

            # Latest productivity per device
            latest_per_device = df_log.sort_values("Timestamp").groupby("device_id").tail(1)

            st.subheader("📌 Current Status by Device")
            st.dataframe(latest_per_device[[
                "device_id", "PredictedProductivity", "RoleName", "RoleGroup", "ProductivityStatus", "Timestamp"
            ]].reset_index(drop=True), use_container_width=True)

            # Avg productivity by device
            st.subheader("📊 Average Productivity by Device")
            avg_prod = df_log.groupby("device_id")["PredictedProductivity"].mean().sort_values(ascending=False)
            st.bar_chart(avg_prod)

            # Role distribution
            st.subheader("👔 Role Distribution Across Devices")
            st.bar_chart(df_log["RoleName"].value_counts())

            # Device-specific timelines
            device_choice = st.selectbox("Select a device to view details", df_log["device_id"].unique())
            device_data = df_log[df_log["device_id"] == device_choice].set_index("Timestamp").sort_index()

            if "PredictedProductivity" in device_data.columns:
                st.subheader(f"📈 Productivity Trend — {device_choice}")
                st.line_chart(device_data["PredictedProductivity"].astype(float))

            if "task_switch_total" in device_data.columns:
                st.subheader(f"🔄 Task Switching — {device_choice}")
                st.line_chart(device_data["task_switch_total"].astype(int))

    except Exception as e:
        st.error(f"❌ Error in Manager View: {e}")

# ================= OTHER MODES =================
elif mode == "📂 Test Dataset Prediction":
    st.subheader("📂 Predictions on Test Dataset")
    # same as before

elif mode == "📊 Feature Importance":
    st.subheader("📊 Feature Importance (Explainability)")
    # same as before
