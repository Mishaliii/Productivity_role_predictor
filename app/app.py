# app/app.py
import os
import sys
import time
import subprocess
import socket
import pandas as pd
import streamlit as st
import joblib
import requests
from datetime import datetime as dt
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ======== PATH FIX ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======== AUTH & DB ========
from auth import show_login
from script.db_manager import (
    insert_log, get_logs, init_db, add_user,
    get_conn, _hash_password
)
from script.collect_realtime_data import get_realtime_data

# ======== INIT DB ========
init_db()

# ======== AUTO-START FLASK SERVER (BEST-EFFORT) ========
def is_port_in_use(host: str, port: int) -> bool:
    """Return True if host:port accepts connections."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def start_flask_server_if_missing(host: str = "127.0.0.1", port: int = 5000):
    """Try to start server/server/app.py if not already running."""
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", "app.py"))
    if not os.path.exists(server_path):
        return
    if is_port_in_use(host, port):
        return
    try:
        subprocess.Popen([sys.executable, server_path],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         close_fds=True)
        time.sleep(1.2)
    except Exception:
        return

start_flask_server_if_missing()

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 5000))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ======== CONFIG ========
PRODUCTIVITY_THRESHOLD = 70
SESSION_ID = int(time.time())

# ======== CACHED MODELS ========
@st.cache_resource
def load_models_safe():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reg_model = joblib.load(os.path.join(base, "models", "regression_model.pkl"))
    cluster_model = joblib.load(os.path.join(base, "models", "clustering_model.pkl"))
    scaler = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    train_feature_list = joblib.load(os.path.join(base, "models", "train_feature_list.pkl"))
    cluster_role_map = joblib.load(os.path.join(base, "models", "cluster_role_map.pkl"))
    return reg_model, cluster_model, scaler, train_feature_list, cluster_role_map

# ======== HELPERS ========
def format_time(seconds: float) -> str:
    try:
        seconds = int(float(seconds))
    except Exception:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    return f"{seconds//3600}h {(seconds%3600)//60}m"

def safe_get(obj, key, default=0):
    try:
        return obj.get(key, default)
    except Exception:
        try:
            return obj[key]
        except Exception:
            return default

def ensure_df_timestamps(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    df = df.copy()
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass
    return df

# ======== LOGIN ========
if not show_login():
    st.stop()

# Sidebar logout
with st.sidebar:
    if st.button("🚪 Logout"):
        for key in ["logged_in", "username", "user_id", "role"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

user = {
    "id": st.session_state.user_id,
    "username": st.session_state.username,
    "role": st.session_state.role,
}

# ======== USER VIEW ========
if user["role"] == "user":
    try:
        reg_model, cluster_model, scaler, train_feature_list, cluster_role_map = load_models_safe()
    except Exception as e:
        st.error(f"❌ Error loading models (required for user view): {e}")
        st.stop()

    st.header(f"⚡ My Productivity Dashboard — {user['username']}")
    st_autorefresh(interval=5000, key="user_realtime")

    try:
        latest_row = get_realtime_data()
        if isinstance(latest_row, pd.DataFrame):
            latest_row = latest_row.iloc[0]

        for col in train_feature_list:
            if col not in latest_row.index:
                latest_row[col] = 0.0

        X = pd.DataFrame([latest_row.loc[train_feature_list].astype(float).to_list()],
                         columns=train_feature_list)
        X_scaled = scaler.transform(X)

        predicted_productivity = round(float(reg_model.predict(X_scaled)[0]), 2)
        predicted_cluster = int(cluster_model.predict(X_scaled)[0])
        predicted_role = cluster_role_map.get(predicted_cluster, f"Cluster {predicted_cluster}")

        role_group_map = {
            "Developer": "Technical",
            "Engineer": "Technical",
            "Designer": "Creative",
            "Writer": "Creative",
            "Manager": "Business",
            "Analyst": "Business",
        }
        role_group = role_group_map.get(predicted_role, predicted_role)
        status = "✅ Productive" if predicted_productivity >= PRODUCTIVITY_THRESHOLD else "❌ Not Productive"

        coding_time = safe_get(latest_row, "time_coding", 0)
        browsing_time = safe_get(latest_row, "time_browsing", 0)
        other_time = safe_get(latest_row, "time_other", 0)

        save_row = latest_row.to_dict()
        save_row.update({
            "PredictedProductivity": predicted_productivity,
            "ClusterID": predicted_cluster,
            "RoleName": predicted_role,
            "RoleGroup": role_group,
            "ProductivityStatus": status,
            "Timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "SessionID": SESSION_ID
        })

        insert_log(user["id"], save_row)
        try:
            requests.post(f"{SERVER_URL}/api/logs",
                          json={**save_row, "user_id": user["id"], "username": user["username"]},
                          timeout=2.5)
        except Exception:
            pass

        df_log = get_logs(user_id=user["id"])
        df_log = ensure_df_timestamps(df_log)

        st.subheader("🧩 Latest Capture + Prediction")
        st.write({
            "PredictedProductivity": predicted_productivity,
            "Role": predicted_role,
            "RoleGroup": role_group,
            "Status": status,
            "Timestamp": save_row["Timestamp"]
        })

        session_avg = (df_log["productivity"].astype(float).mean()
                       if (df_log is not None and not df_log.empty and "productivity" in df_log.columns)
                       else predicted_productivity)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Latest Productivity", f"{predicted_productivity:.2f}")
        c2.metric("Session Avg", f"{session_avg:.2f}")
        c3.metric("Role", predicted_role)
        c4.metric("Group", role_group)
        c5.metric("Status", status)
        c6.metric("Coding Time", format_time(coding_time))
        c7.metric("Browsing Time", format_time(browsing_time))
        c8.metric("Other Time", format_time(other_time))

        st.subheader("📌 My Recent Logs")
        if df_log is None or df_log.empty:
            st.info("No logs saved yet.")
        else:
            st.dataframe(df_log.head(15).reset_index(drop=True), use_container_width=True)

        latest_features = {
            "Typing": float(safe_get(latest_row, "typing_intensity", 0)),
            "Mouse": float(safe_get(latest_row, "mouse_intensity", 0)),
            "Idle": float(safe_get(latest_row, "idle_duration", 0)),
            "Switches": float(safe_get(latest_row, "task_switch_total", 0)),
        }
        radar_df = pd.DataFrame({"Feature": list(latest_features.keys()), "Value": list(latest_features.values())})
        fig_radar = px.line_polar(radar_df, r="Value", theta="Feature", line_close=True, title="Latest Feature Mix")
        fig_radar.update_traces(fill="toself")
        st.plotly_chart(fig_radar, use_container_width=True)

        if df_log is not None and not df_log.empty and "timestamp" in df_log.columns:
            if "productivity" in df_log.columns:
                st.subheader("📈 Productivity Trend")
                st.line_chart(df_log.sort_values("timestamp").set_index("timestamp")["productivity"].astype(float))
            if "task_switch_total" in df_log.columns:
                st.subheader("🔄 Task Switching Trend")
                st.line_chart(df_log.sort_values("timestamp").set_index("timestamp")["task_switch_total"].astype(int))
            if "idle_duration" in df_log.columns:
                st.subheader("⏳ Idle Duration Trend")
                st.line_chart(df_log.sort_values("timestamp").set_index("timestamp")["idle_duration"].astype(float))

    except Exception as e:
        st.error(f"❌ Error in personal dashboard: {e}")

# ======== ADMIN VIEW ========
elif user["role"] == "admin":
    st.sidebar.title("👔 Admin Panel")
    refresh_option = st.sidebar.selectbox("Auto-refresh interval", ["Off", "5s", "15s", "30s"], index=1)
    interval_map = {"Off": 0, "5s": 5, "15s": 15, "30s": 30}
    refresh_seconds = interval_map.get(refresh_option, 5)
    if refresh_seconds > 0:
        st_autorefresh(interval=refresh_seconds * 1000, key=f"admin_realtime_{refresh_seconds}")

    admin_option = st.sidebar.radio(
        "Choose what to view",
        ["Users Data", "Loaded Dataset", "Self Productivity Testing", "User Management"]
    )

    if admin_option == "Users Data":
        st.header("👥 Users Overview (Admin)")
        try:
            df_log = get_logs()
            if df_log is None or df_log.empty:
                st.warning("⚠️ No logs yet.")
            else:
                df_log = ensure_df_timestamps(df_log)
                latest_per_user = df_log.sort_values("timestamp").groupby("user_id", as_index=False).last()
                avg_prod = df_log.groupby("user_id", as_index=False)["productivity"].mean()
                avg_prod.columns = ["user_id", "avg_productivity"]
                summary = latest_per_user.merge(avg_prod, on="user_id", how="left")

                now = dt.now()
                summary["LastActive"] = pd.to_datetime(summary["timestamp"], errors="coerce")
                summary["active_status"] = summary["LastActive"].apply(
                    lambda t: "🟢 Active" if (pd.notnull(t) and (now - t).total_seconds() < 120) else "🔴 Inactive"
                )

                if "username" not in summary.columns:
                    summary["username"] = summary["user_id"].apply(lambda x: f"User {x}")

                st.subheader("📌 Users at a Glance")
                display_table = summary[["username", "role", "avg_productivity", "timestamp", "status", "active_status"]]
                display_table = display_table.sort_values("avg_productivity", ascending=False).reset_index(drop=True)
                st.dataframe(display_table, use_container_width=True)

                st.subheader("📊 Leaderboard — Average Productivity")
                leaderboard = summary.sort_values("avg_productivity", ascending=False)
                fig_lb = px.bar(
                    leaderboard,
                    x="username",
                    y="avg_productivity",
                    color="role",
                    text=leaderboard["avg_productivity"].round(2),
                    title="Average Productivity by User"
                )
                fig_lb.update_traces(textposition="outside")
                st.plotly_chart(fig_lb, use_container_width=True)

                st.subheader("🔎 Team Productivity Status")
                status_counts = summary["status"].fillna("Unknown").value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig_status = px.pie(status_counts, names="status", values="count", hole=0.4)
                st.plotly_chart(fig_status, use_container_width=True)

                st.subheader("👥 Role Distribution")
                role_counts = summary["role"].fillna("Unknown").value_counts().reset_index()
                role_counts.columns = ["role", "count"]
                fig_roles = px.bar(role_counts, x="role", y="count", text="count", title="Users by Role")
                fig_roles.update_traces(textposition="outside")
                st.plotly_chart(fig_roles, use_container_width=True)

                st.markdown("---")
                st.subheader("🔍 Inspect a User (drilldown)")
                usernames = sorted(summary["username"].unique().tolist())
                if usernames:
                    selected_user = st.selectbox("Select user to inspect", usernames, index=0)
                    df_log = get_logs()
                    df_log = ensure_df_timestamps(df_log)
                    user_data = df_log[df_log["username"] == selected_user].sort_values("timestamp")

                    if user_data.empty and "user_id" in df_log.columns:
                        user_ids = sorted(df_log["user_id"].unique().tolist())
                        selected_id = st.selectbox("Or select user_id", user_ids, index=0)
                        user_data = df_log[df_log["user_id"] == selected_id].sort_values("timestamp")
                        if "username" in df_log.columns and not user_data.empty:
                            selected_user = user_data["username"].iloc[-1]

                    if user_data.empty:
                        st.info("No logs for selected user.")
                    else:
                        st.write(f"### Latest logs for {selected_user}")
                        st.dataframe(user_data.tail(20).reset_index(drop=True), use_container_width=True)
                        if "productivity" in user_data.columns:
                            st.subheader("📈 Productivity Trend")
                            st.line_chart(user_data.set_index("timestamp")["productivity"].astype(float))
                        if "task_switch_total" in user_data.columns:
                            st.subheader("🔄 Task Switching Trend")
                            st.line_chart(user_data.set_index("timestamp")["task_switch_total"].astype(int))
                        if "idle_duration" in user_data.columns:
                            st.subheader("⏳ Idle Duration Trend")
                            st.line_chart(user_data.set_index("timestamp")["idle_duration"].astype(float))
                else:
                    st.info("No users to inspect.")
        except Exception as e:
            st.error(f"❌ Error in admin dashboard: {e}")

    elif admin_option == "Loaded Dataset":
        st.header("📂 Preview Dataset")
        dataset_choice = st.selectbox("Select dataset", ["Train", "Test"])
        file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data",
                                 "train.csv" if dataset_choice == "Train" else "test.csv")
        if os.path.exists(file_path):
            df_data = pd.read_csv(file_path)
            st.dataframe(df_data.head(50), use_container_width=True)
            st.write("Shape:", df_data.shape)
        else:
            st.error(f"Dataset not found: {file_path}")

    elif admin_option == "Self Productivity Testing":
        st.header("🧪 Admin Self Productivity Test")
        try:
            reg_model, cluster_model, scaler, train_feature_list, cluster_role_map = load_models_safe()
            latest_row = get_realtime_data()
            if isinstance(latest_row, pd.DataFrame):
                latest_row = latest_row.iloc[0]
            for col in train_feature_list:
                if col not in latest_row.index:
                    latest_row[col] = 0
            X = pd.DataFrame([latest_row.loc[train_feature_list].astype(float).to_list()],
                             columns=train_feature_list)
            X_scaled = scaler.transform(X)
            predicted_productivity = round(float(reg_model.predict(X_scaled)[0]), 2)
            predicted_cluster = int(cluster_model.predict(X_scaled)[0])
            predicted_role = cluster_role_map.get(predicted_cluster, f"Cluster {predicted_cluster}")
            status = "✅ Productive" if predicted_productivity >= PRODUCTIVITY_THRESHOLD else "❌ Not Productive"

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Productivity", f"{predicted_productivity:.2f}")
            c2.metric("Predicted Role", predicted_role)
            c3.metric("Status", status)

            st.subheader("Raw Capture (not saved)")
            st.write(latest_row.to_dict())
        except Exception as e:
            st.error(f"❌ Error in self test: {e}")

    elif admin_option == "User Management":
        st.header("🔐 Admin — User Management")
        st.subheader("➕ Create User")
        with st.form("create_user_form"):
            new_username = st.text_input("New user's username")
            new_password = st.text_input("New user's password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            submitted = st.form_submit_button("Create user")
            if submitted:
                if not new_username or not new_password:
                    st.error("❌ Please enter username and password.")
                else:
                    created = add_user(new_username, new_password, role=new_role)
                    if created:
                        st.success(f"✅ User '{new_username}' created with role '{new_role}'.")
                        st.experimental_rerun()
                    else:
                        st.error("⚠️ Username already exists.")

        st.markdown("---")
        st.subheader("🔑 Reset User Password")
        with st.form("reset_password_form"):
            with get_conn() as conn:
                users_df = pd.read_sql_query("SELECT username FROM users", conn)
            usernames = users_df["username"].tolist() if not users_df.empty else []
            if usernames:
                selected_user = st.selectbox("Select user", usernames, key="reset_user_select")
                new_password_admin = st.text_input("New password", type="password", key="reset_user_pass")
                submitted_reset = st.form_submit_button("Reset Password")
                if submitted_reset:
                    if not new_password_admin:
                        st.error("❌ Please enter a new password.")
                    else:
                        new_hash, new_salt = _hash_password(new_password_admin)
                        with get_conn() as conn:
                            conn.execute(
                                "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                                (new_hash, new_salt, selected_user),
                            )
                            conn.commit()
                        st.success(f"✅ Password for '{selected_user}' has been reset.")
                        st.experimental_rerun()
            else:
                st.warning("⚠️ No users found in database.")
