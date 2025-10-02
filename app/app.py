# app/app.py
import os
import sys
import time
import threading
import subprocess
import socket
from datetime import datetime as dt

import pandas as pd
import streamlit as st
import joblib
import requests
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ===== PATH FIX =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===== AUTH & DB =====
from auth import show_login
from script.db_manager import (
    insert_log, get_logs, init_db, add_user,
    get_conn, _hash_password
)
from script.collect_realtime_data import get_realtime_data

# ===== INIT DB =====
try:
    init_db()
except Exception as e:
    st.error(f"❌ Database init failed: {e}")

# ===== AUTO-START FLASK SERVER (optional) =====
def is_port_in_use(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def start_flask_server_if_missing(host: str = "127.0.0.1", port: int = 5000):
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", "app.py"))
    if not os.path.exists(server_path):
        return
    if is_port_in_use(host, port):
        return
    try:
        subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True
        )
        time.sleep(1.2)
    except Exception:
        return

start_flask_server_if_missing()

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 5000))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ===== CONFIG =====
PRODUCTIVITY_THRESHOLD = 70
SESSION_ID = int(time.time())
COLLECT_INTERVAL_SEC = 5

# ===== LOAD MODELS (cached) =====
@st.cache_resource(show_spinner=False)
def load_models_safe():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths = {
        "reg": os.path.join(base, "models", "regression_model.pkl"),
        "cluster": os.path.join(base, "models", "clustering_model.pkl"),
        "scaler": os.path.join(base, "models", "scaler.pkl"),
        "features": os.path.join(base, "models", "train_feature_list.pkl"),
        "rolemap": os.path.join(base, "models", "cluster_role_map.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)} in models/")
    reg_model = joblib.load(paths["reg"])
    cluster_model = joblib.load(paths["cluster"])
    scaler = joblib.load(paths["scaler"])
    train_feature_list = joblib.load(paths["features"])
    cluster_role_map = joblib.load(paths["rolemap"])
    return reg_model, cluster_model, scaler, train_feature_list, cluster_role_map

# ===== HELPERS =====
def safe_get(obj, key, default=0):
    try:
        return obj.get(key, default)
    except Exception:
        try:
            return obj[key]
        except Exception:
            return default

def unify_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "timestamp" not in df.columns and "Timestamp" in df.columns:
        df["timestamp"] = df["Timestamp"]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def get_prod_col(df: pd.DataFrame) -> str:
    if df is not None and not df.empty:
        if "PredictedProductivity" in df.columns:
            return "PredictedProductivity"
        if "productivity" in df.columns:
            return "productivity"
    return "PredictedProductivity"

def get_status_col(df: pd.DataFrame) -> str:
    if df is not None and not df.empty:
        if "ProductivityStatus" in df.columns:
            return "ProductivityStatus"
        if "status" in df.columns:
            return "status"
    return "ProductivityStatus"

ROLE_GROUP_MAP = {
    "Developer": "Technical",
    "Engineer": "Technical",
    "Designer": "Creative",
    "Writer": "Creative",
    "Manager": "Business",
    "Analyst": "Business",
    "Idle": "Idle",
}

def predict_and_build_row(latest_row: pd.Series,
                          reg_model, cluster_model, scaler, train_feature_list, cluster_role_map) -> dict:
    latest_row = latest_row.copy()
    for col in train_feature_list:
        if col not in latest_row.index:
            latest_row[col] = 0.0
    X = pd.DataFrame(
        [latest_row.loc[train_feature_list].astype(float).to_list()],
        columns=train_feature_list
    )
    X_scaled = scaler.transform(X)

    predicted_productivity = round(float(reg_model.predict(X_scaled)[0]), 2)
    predicted_cluster = int(cluster_model.predict(X_scaled)[0])
    predicted_role = cluster_role_map.get(predicted_cluster, f"Cluster {predicted_cluster}")

    coding_time = float(safe_get(latest_row, "time_coding", 0))
    browsing_time = float(safe_get(latest_row, "time_browsing", 0))
    other_time = float(safe_get(latest_row, "time_other", 0))
    if (coding_time + browsing_time + other_time) == 0:
        predicted_productivity = 0.0
        predicted_role = "Idle"

    role_group = ROLE_GROUP_MAP.get(predicted_role, predicted_role)
    status = "✅ Productive" if predicted_productivity >= PRODUCTIVITY_THRESHOLD else "❌ Not Productive"
    now_str = dt.now().strftime("%Y-%m-%d %H:%M:%S")

    save_row = latest_row.to_dict()
    save_row.update({
        "PredictedProductivity": predicted_productivity,
        "productivity": predicted_productivity,
        "ClusterID": predicted_cluster,
        "RoleName": predicted_role,
        "RoleGroup": role_group,
        "ProductivityStatus": status,
        "status": status,
        "Timestamp": now_str,
        "timestamp": now_str,
        "SessionID": SESSION_ID
    })
    return save_row

# ===== FRESH LOG FETCH (used instead of st.cache) =====
def get_logs_fresh(user_id: int = None, limit: int = None) -> pd.DataFrame:
    """Always fetch latest logs directly from DB (no caching)."""
    try:
        with get_conn() as conn:
            query = "SELECT * FROM logs"
            params = []
            if user_id is not None:
                query += " WHERE user_id=?"
                params.append(user_id)
            query += " ORDER BY datetime(timestamp) DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        print("[get_logs_fresh] error:", e)
        return pd.DataFrame()


# ===== LOGIN =====
if not show_login():
    st.stop()

user = {
    "id": st.session_state.user_id,
    "username": st.session_state.username,
    "role": st.session_state.role,
}

# ===== BACKGROUND COLLECTOR =====
def ensure_background_collector(user_ctx: dict):
    if st.session_state.get("collector_running", False):
        print("[collector] already running")
        return

    print("[collector] starting background thread...")
    try:
        reg_model, cluster_model, scaler, train_feature_list, cluster_role_map = load_models_safe()
    except Exception as e:
        st.error(f"❌ Could not load models for collector: {e}")
        return

    def collector_loop(user_id: int, username: str):
        print(f"[collector_loop] started for user {username} ({user_id})")
        while True:
            try:
                latest_df = get_realtime_data()
                print("[collector_loop] realtime data:", latest_df)
                if latest_df is None:
                    time.sleep(COLLECT_INTERVAL_SEC)
                    continue

                if isinstance(latest_df, pd.DataFrame) and not latest_df.empty:
                    latest_row = latest_df.iloc[0]
                elif isinstance(latest_df, pd.Series):
                    latest_row = latest_df
                else:
                    time.sleep(COLLECT_INTERVAL_SEC)
                    continue

                row = predict_and_build_row(
                    latest_row, reg_model, cluster_model, scaler, train_feature_list, cluster_role_map
                )
                print("[collector_loop] inserting row:", row)
                insert_log(user_id, row)

            except Exception as e:
                print(f"[collector_loop] error: {e}")

            time.sleep(COLLECT_INTERVAL_SEC)

    t = threading.Thread(target=collector_loop, args=(user_ctx["id"], user_ctx["username"]), daemon=True)
    t.start()
    st.session_state["collector_running"] = True
    print("[collector] thread started!")


# ================= USER VIEW =================
if user["role"] == "user":
    ensure_background_collector(user)
    st.header(f"⚡ My Productivity Dashboard — {user['username']}")
    st_autorefresh(interval=5000, key="user_refresh")

    try:
        with st.spinner("Collecting activity and updating predictions..."):
            latest_df = get_realtime_data()
            latest_row = latest_df.iloc[0] if isinstance(latest_df, pd.DataFrame) and not latest_df.empty else latest_df
            if latest_row is None:
                st.warning("No realtime data captured yet. Keep the app running…")
                st.stop()

            reg_model, cluster_model, scaler, train_feature_list, cluster_role_map = load_models_safe()
            save_row = predict_and_build_row(latest_row, reg_model, cluster_model, scaler, train_feature_list, cluster_role_map)

        df_log = get_logs_fresh(user_id=user["id"])
        df_log = unify_timestamps(df_log)
        prod_col = get_prod_col(df_log)

        st.subheader("🧩 Latest Capture & Prediction")
        k1, k2, k3 = st.columns(3)
        k1.metric("Productivity", f"{save_row.get('PredictedProductivity', 0):.2f}")
        k2.metric("Role", save_row.get("RoleName", 'Unknown'))
        k3.metric("Status", save_row.get("ProductivityStatus", 'Unknown'))
        k4, k5, k6 = st.columns(3)
        total_active = float(safe_get(latest_row, "time_coding", 0)) + float(safe_get(latest_row, "time_browsing", 0)) + float(safe_get(latest_row, "time_other", 0))
        k4.metric("Group", save_row.get("RoleGroup", 'Unknown'))
        k5.metric("Active Time (s)", int(total_active))
        k6.metric("Last Update", save_row.get("Timestamp", ''))

        session_avg = float(df_log[prod_col].astype(float).mean()) if (df_log is not None and not df_log.empty and prod_col in df_log.columns) else float(save_row.get("PredictedProductivity", 0))
        c1, c2 = st.columns(2)
        c1.metric("Session Avg", f"{session_avg:.2f}")
        rows_to_show = c2.selectbox("Rows to show", [20, 50, 100, 200, 500], index=1)

        st.subheader("📌 My Recent Logs")
        if df_log is None or df_log.empty:
            st.info("No logs saved yet.")
        else:
            df_show = df_log.sort_values("timestamp", ascending=False)
            st.dataframe(df_show.head(rows_to_show).reset_index(drop=True), use_container_width=True)

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
            df_sorted = df_log.sort_values("timestamp")
            if prod_col in df_sorted.columns:
                st.subheader("📈 Productivity Trend")
                st.line_chart(df_sorted.set_index("timestamp")[prod_col].astype(float))
            if "task_switch_total" in df_sorted.columns:
                st.subheader("🔄 Task Switching Trend")
                st.line_chart(df_sorted.set_index("timestamp")["task_switch_total"].astype("Int64"))
            if "idle_duration" in df_sorted.columns:
                st.subheader("⏳ Idle Duration Trend")
                st.line_chart(df_sorted.set_index("timestamp")["idle_duration"].astype(float))

        # ===== 🚪 Logout Button =====
        if st.button("🚪 Logout", key="logout_user"):
            try:
                with get_conn() as conn:
                    # Insert a logout marker log (so admin sees inactive immediately)
                    conn.execute(
                        """
                        INSERT INTO logs (user_id, username, timestamp, status, productivity)
                        VALUES (?, ?, datetime('now'), ?, ?)
                        """,
                        (user["id"], user["username"], "🚪 Logged Out", 0),
                    )
                    conn.commit()
            except Exception as e:
                print("[logout] Failed to mark inactive:", e)

            # Stop background collector
            st.session_state["collector_running"] = False

            # Clear session and refresh page
            st.session_state.clear()
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error in personal dashboard: {e}")


## ================= ADMIN VIEW =================
elif user["role"] == "admin":
    st.sidebar.title("👔 Admin Panel")
    refresh_option = st.sidebar.selectbox("Auto-refresh interval", ["Off", "5s", "15s", "30s"], index=1)
    interval_map = {"Off": 0, "5s": 5, "15s": 15, "30s": 30}
    refresh_seconds = interval_map.get(refresh_option, 5)
    if refresh_seconds > 0:
        st_autorefresh(interval=refresh_seconds * 1000, key=f"admin_refresh_{refresh_seconds}")
    if st.sidebar.button("🔄 Manual Refresh", key="admin_manual_refresh"):
        st.rerun()

    admin_option = st.sidebar.radio(
        "Choose what to view",
        ["Users Data", "Loaded Dataset", "Self Productivity Testing", "User Management"]
    )

    # ---- Users Data ----
    if admin_option == "Users Data":
        st.header("👥 Users Overview (Admin)")
        try:
            df_log = get_logs_fresh()
            print("DEBUG → Admin fetched logs:", len(df_log))

            if df_log is None or df_log.empty:
                st.warning("⚠️ No logs yet.")
            else:
                print("DEBUG → Latest 5 timestamps:\n", df_log["timestamp"].head())
                df_log = unify_timestamps(df_log)

                prod_col = "productivity" if "productivity" in df_log.columns else "PredictedProductivity"
                status_col = "status" if "status" in df_log.columns else (
                    "ProductivityStatus" if "ProductivityStatus" in df_log.columns else None
                )
                print("DEBUG → prod_col used:", prod_col)

                df_log[prod_col] = pd.to_numeric(df_log[prod_col], errors="coerce")

                # --- Latest record for each user ---
                latest_per_user = df_log.sort_values("timestamp").groupby("user_id", as_index=False).last()

                # --- Average productivity ---
                avg_prod = df_log.groupby("user_id", as_index=False)[prod_col].mean()
                avg_prod.rename(columns={prod_col: "avg_productivity"}, inplace=True)
                summary = latest_per_user.merge(avg_prod, on="user_id", how="left")
                summary["avg_productivity"] = summary["avg_productivity"].fillna(0).round(2)

                # --- Active status ---
                now = pd.Timestamp.now()
                active_window = max(10, refresh_seconds * 2 + 5)
                summary["LastActive"] = pd.to_datetime(summary["timestamp"], errors="coerce")

                if "status" in summary.columns:
                    is_logged_out = summary["status"].astype(str).str.strip().eq("🚪 Logged Out")
                else:
                    is_logged_out = pd.Series(False, index=summary.index)

                is_recent = summary["LastActive"].notna() & (
                    (now - summary["LastActive"]).dt.total_seconds() < active_window
                )

                summary["active_status"] = "🔴 Inactive"
                summary.loc[is_recent & (~is_logged_out), "active_status"] = "🟢 Active"

                if "username" not in summary.columns:
                    summary["username"] = summary["user_id"].apply(lambda x: f"User {x}")

                st.subheader("📌 Users at a Glance")
                show_cols = ["username", "role", "avg_productivity", "timestamp"]
                if status_col and status_col in summary.columns:
                    show_cols.append(status_col)
                if "active_status" in summary.columns:
                    show_cols.append("active_status")

                st.dataframe(
                    summary[show_cols]
                    .rename(columns={status_col: "status"} if status_col else {})
                    .sort_values("avg_productivity", ascending=False, na_position="last")
                    .reset_index(drop=True),
                    use_container_width=True
                )

                st.subheader("📊 Leaderboard — Average Productivity")
                fig_lb = px.bar(
                    summary.sort_values("avg_productivity", ascending=False, na_position="last"),
                    x="username", y="avg_productivity", color="role",
                    text=summary["avg_productivity"],
                    title="Average Productivity by User"
                )
                fig_lb.update_traces(textposition="outside")
                st.plotly_chart(fig_lb, use_container_width=True)

                if status_col and status_col in summary.columns:
                    st.subheader("🔎 Team Productivity Status")
                    status_counts = summary[status_col].fillna("Unknown").value_counts().reset_index()
                    status_counts.columns = ["status", "count"]
                    fig_status = px.pie(status_counts, names="status", values="count", hole=0.4)
                    st.plotly_chart(fig_status, use_container_width=True)

                if "role" in summary.columns:
                    st.subheader("👥 Role Distribution")
                    role_counts = summary["role"].fillna("Unknown").value_counts().reset_index()
                    role_counts.columns = ["role", "count"]
                    fig_roles = px.bar(role_counts, x="role", y="count", text="count", title="Users by Role")
                    fig_roles.update_traces(textposition="outside")
                    st.plotly_chart(fig_roles, use_container_width=True)

                st.markdown("---")
                st.subheader("🔍 Inspect a User (drilldown)")
                usernames = sorted(summary["username"].dropna().unique().tolist())
                if usernames:
                    selected_user = st.selectbox("Select user to inspect", usernames, index=0)
                    selected_user_id = summary.loc[summary["username"] == selected_user, "user_id"].iloc[0]

                    df_all = get_logs_fresh()
                    df_all = unify_timestamps(df_all)
                    prod_col = "productivity" if "productivity" in df_all.columns else "PredictedProductivity"
                    user_data = df_all[df_all["user_id"] == selected_user_id].sort_values("timestamp")

                    if user_data.empty:
                        st.info("No logs for selected user.")
                    else:
                        rows_drill = st.selectbox("Rows to show", [20, 50, 100, 200, 500], index=1)
                        st.dataframe(user_data.tail(rows_drill).reset_index(drop=True), use_container_width=True)

                        if prod_col in user_data.columns:
                            st.subheader("📈 Productivity Trend")
                            st.line_chart(user_data.set_index("timestamp")[prod_col].astype(float))
                        if "task_switch_total" in user_data.columns:
                            st.subheader("🔄 Task Switching Trend")
                            st.line_chart(user_data.set_index("timestamp")["task_switch_total"].astype("Int64"))
                        if "idle_duration" in user_data.columns:
                            st.subheader("⏳ Idle Duration Trend")
                            st.line_chart(user_data.set_index("timestamp")["idle_duration"].astype(float))
                else:
                    st.info("No users to inspect.")
        except Exception as e:
            st.error(f"❌ Error in admin dashboard: {e}")

    # ---- Loaded Dataset ----
    elif admin_option == "Loaded Dataset":
        st.header("📂 Loaded Dataset (Training Data)")
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_path = os.path.join(base, "data", "Enhanced_WorkerProductivity_Dataset.xlsx")
            if os.path.exists(data_path):
                df_dataset = pd.read_excel(data_path)
                st.success(f"✅ Loaded dataset from: `{data_path}`")
                st.dataframe(df_dataset.head(100), use_container_width=True)
            else:
                st.warning("⚠️ Dataset file not found.")
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")

    # ---- Self Test ----
    elif admin_option == "Self Productivity Testing":
        st.header("🧪 Admin Self Productivity Test")
        try:
            with st.spinner("Running realtime capture and prediction..."):
                reg_model, cluster_model, scaler, train_feature_list, cluster_role_map = load_models_safe()
                latest_df = get_realtime_data()
                latest_row = latest_df.iloc[0] if isinstance(latest_df, pd.DataFrame) and not latest_df.empty else latest_df
                if latest_row is None:
                    st.warning("No realtime data captured yet.")
                    st.stop()
                save_row = predict_and_build_row(latest_row, reg_model, cluster_model, scaler, train_feature_list, cluster_role_map)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Productivity", f"{save_row.get('PredictedProductivity', 0):.2f}")
            c2.metric("Predicted Role", save_row.get("RoleName", "Unknown"))
            c3.metric("Status", save_row.get("ProductivityStatus", "Unknown"))

            st.subheader("Latest Capture (key fields)")
            show_keys = ["typing_intensity", "mouse_intensity", "idle_duration", "task_switch_total",
                         "time_coding", "time_browsing", "time_other"]
            latest_display = {k: safe_get(latest_row, k, None) for k in show_keys}
            st.table(pd.DataFrame(list(latest_display.items()), columns=["Feature", "Value"]))
        except Exception as e:
            st.error(f"❌ Error in self test: {e}")

    # ---- User Management ----
    elif admin_option == "User Management":
        st.header("🔐 Admin — User Management")
        st.subheader("➕ Create User")
        with st.form("create_user_form"):
            new_username = st.text_input("New user's username")
            new_password = st.text_input("New user's password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            submitted = st.form_submit_button("Create user")
            if submitted:
                if not new_username or not new_password:
                    st.error("❌ Please enter username and password.")
                else:
                    try:
                        created = add_user(new_username, new_password, role=new_role)
                        if created:
                            st.toast(f"✅ User '{new_username}' created with role '{new_role}'.", icon="✅")
                            st.rerun()
                        else:
                            st.error("⚠️ Username already exists.")
                    except Exception as e:
                        st.error(f"❌ Failed to create user: {e}")

        st.markdown("---")
        st.subheader("🔑 Reset User Password")
        with st.form("reset_password_form"):
            try:
                with get_conn() as conn:
                    users_df = pd.read_sql_query("SELECT username FROM users", conn)
            except Exception as e:
                users_df = pd.DataFrame()
                st.error(f"❌ Could not load users: {e}")

            usernames = users_df["username"].tolist() if not users_df.empty else []
            if usernames:
                selected_user = st.selectbox("Select user", usernames, key="reset_user_select")
                new_password_admin = st.text_input("New password", type="password", key="reset_user_pass")
                submitted_reset = st.form_submit_button("Reset Password")
                if submitted_reset:
                    if not new_password_admin:
                        st.error("❌ Please enter a new password.")
                    else:
                        try:
                            new_hash, new_salt = _hash_password(new_password_admin)
                            with get_conn() as conn:
                                conn.execute(
                                    "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                                    (new_hash, new_salt, selected_user),
                                )
                                conn.commit()
                            st.toast(f"✅ Password for '{selected_user}' has been reset.", icon="🔑")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to reset password: {e}")
            else:
                st.warning("⚠️ No users found in database.")
