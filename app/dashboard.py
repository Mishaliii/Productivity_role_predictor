# app/dashboard.py
import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ----------- Paths -----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "usage_log.csv")

st.set_page_config(page_title="📊 Productivity Dashboard", layout="wide")

# ----------- Auto Refresh -----------
st_autorefresh(interval=10 * 1000, key="datarefresh")
st.sidebar.info("⏳ Dashboard refreshes every 10 seconds")

# ----------- Safe CSV Load -----------
@st.cache_data(ttl=10)
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# ----------- Main Dashboard -----------
st.title("⚡ Real-Time Productivity & Role Dashboard")

if df.empty:
    st.warning("No data available yet. Please run `collect_realtime_data.py` to generate data.")
else:
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    st.subheader("📌 Latest Logs")
    st.dataframe(df.tail(10), use_container_width=True)

    if "Predicted_Productivity_Score" in df.columns and "Timestamp" in df.columns:
        st.subheader("📈 Productivity Over Time")
        st.line_chart(df.set_index("Timestamp")["Predicted_Productivity_Score"])

    if "Predicted_Role" in df.columns:
        st.subheader("👥 Role Distribution")
        role_counts = df["Predicted_Role"].value_counts()
        st.bar_chart(role_counts)

    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))
