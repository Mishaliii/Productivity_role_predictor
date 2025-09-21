# app/app.py
import os
import sys
import time
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime as dt
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ======== PATH FIX ========
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ======== AUTH & DB ========
from auth import show_login
from script.db_manager import (
    insert_log, get_logs, init_db, add_user,
    change_password, get_conn, _hash_password
)
from script.collect_realtime_data import get_realtime_data

# Ensure DB exists
init_db()

# Require login before showing the app
if not show_login():
    st.stop()

# Current logged-in user from session_state
user = {
    "id": st.session_state.user_id,
    "username": st.session_state.username,
    "role": st.session_state.role,
}

# ======== CONFIG ========
PRODUCTIVITY_THRESHOLD = 70
SESSION_ID = int(time.time())

# ======== HELPERS ========
def format_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        return f"{seconds//3600}h {(seconds%3600)//60}m"


# ======== USER DASHBOARD ========
if user["role"] == "user":
    try:
        reg_model = joblib.load("models/regression_model.pkl")
        cluster_model = joblib.load("models/clustering_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        train_feature_list = joblib.load("models/train_feature_list.pkl")
        cluster_role_map = joblib.load("models/cluster_role_map.pkl")
    except Exception as e:
        st.error(f"❌ Error loading models (required for user view): {e}")
        st.stop()

    st.header(f"⚡ My Productivity Dashboard — {user['username']}")

    st_autorefresh(interval=5000, key="realtime_refresh")

    try:
        latest_row = get_realtime_data()

        for col in train_feature_list:
            if col not in latest_row.columns:
                latest_row[col] = 0
        X = latest_row[train_feature_list].astype(float)
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

        coding_time = latest_row.get("time_coding", pd.Series([0])).iloc[0]
        browsing_time = latest_row.get("time_browsing", pd.Series([0])).iloc[0]
        other_time = latest_row.get("time_other", pd.Series([0])).iloc[0]

        save_row = latest_row.copy().reset_index(drop=True)
        save_row["PredictedProductivity"] = predicted_productivity
        save_row["ClusterID"] = predicted_cluster
        save_row["RoleName"] = predicted_role
        save_row["RoleGroup"] = role_group
        save_row["ProductivityStatus"] = status
        save_row["Timestamp"] = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        save_row["SessionID"] = SESSION_ID

        # User: insert into logs
        insert_log(user["id"], save_row.iloc[0].to_dict())

        df_log = get_logs(user_id=user["id"])

        st.subheader("🧩 Latest Capture + Prediction")
        st.write(save_row)

        session_avg = (
            df_log["productivity"].astype(float).mean()
            if ("productivity" in df_log.columns and not df_log.empty)
            else predicted_productivity
        )

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("Latest Productivity", f"{predicted_productivity:.2f}")
        col2.metric("Session Avg", f"{session_avg:.2f}")
        col3.metric("Role", predicted_role)
        col4.metric("Group", role_group)
        col5.metric("Status", status)
        col6.metric("Coding Time", format_time(coding_time))
        col7.metric("Browsing Time", format_time(browsing_time))
        col8.metric("Other Time", format_time(other_time))

        st.subheader("📌 My Recent Logs")
        if df_log.empty:
            st.write("No logs yet for this user.")
        else:
            st.dataframe(df_log.head(15), use_container_width=True)

        latest_features = {
            "Typing": float(latest_row.get("typing_intensity", pd.Series([0])).iloc[0]),
            "Mouse": float(latest_row.get("mouse_intensity", pd.Series([0])).iloc[0]),
            "Idle": float(latest_row.get("idle_duration", pd.Series([0])).iloc[0]),
            "Switches": float(latest_row.get("task_switch_total", pd.Series([0])).iloc[0]),
        }
        radar_df = pd.DataFrame({"Feature": list(latest_features.keys()), "Value": list(latest_features.values())})
        fig = px.line_polar(radar_df, r="Value", theta="Feature", line_close=True)
        fig.update_traces(fill="toself")
        st.subheader("📊 Feature Contribution (Latest Capture)")
        st.plotly_chart(fig, use_container_width=True)

        if not df_log.empty:
            if "timestamp" in df_log.columns and "productivity" in df_log.columns:
                try:
                    st.subheader("📈 Productivity Trend")
                    st.line_chart(df_log.set_index("timestamp")["productivity"].astype(float))
                except Exception:
                    pass

            if "task_switch_total" in df_log.columns:
                st.subheader("🔄 Task Switching (Cumulative)")
                st.line_chart(df_log.set_index("timestamp")["task_switch_total"].astype(int))

            if "idle_duration" in df_log.columns:
                st.subheader("⏳ Idle Duration Trend")
                st.line_chart(df_log.set_index("timestamp")["idle_duration"].astype(float))

    except Exception as e:
        st.error(f"❌ Error in personal dashboard: {e}")


# ======== ADMIN DASHBOARD ========
elif user["role"] == "admin":
    st.sidebar.title("👔 Admin Panel")
    admin_option = st.sidebar.radio(
        "Choose what to view",
        ["Users Data", "Loaded Dataset", "Self Productivity Testing"]
    )

    # --- Option 1: Users Data ---
    if admin_option == "Users Data":
        st.header("👥 Users Overview")
        try:
            df_log = get_logs()

            if df_log.empty:
                st.warning("⚠️ No logs yet. Users must generate activity.")
            else:
                latest_per_user = df_log.sort_values("timestamp").groupby("user_id", as_index=False).last()
                avg_prod = df_log.groupby("user_id", as_index=False)["productivity"].mean()
                avg_prod.columns = ["user_id", "avg_productivity"]

                summary = latest_per_user.merge(avg_prod, on="user_id", how="left")

                now = dt.now()
                summary["LastActive"] = pd.to_datetime(summary["timestamp"], errors="coerce")
                summary["active_status"] = summary["LastActive"].apply(
                    lambda t: "🟢" if (pd.notnull(t) and (now - t).total_seconds() < 120) else "🔴"
                )

                if "username" in summary.columns:
                    summary["display_name"] = summary["active_status"] + " " + summary["username"]
                    show_cols = ["display_name", "role", "avg_productivity", "timestamp", "status"]
                else:
                    summary["display_name"] = summary["active_status"] + " User " + summary["user_id"].astype(str)
                    show_cols = ["display_name", "role", "avg_productivity", "timestamp", "status"]

                st.subheader("📌 Users at a Glance")
                st.dataframe(
                    summary[show_cols].sort_values("avg_productivity", ascending=False).reset_index(drop=True),
                    use_container_width=True
                )

                # --- Leaderboard ---
                st.subheader("📊 Leaderboard — Average Productivity")
                label_col = "username" if "username" in summary.columns else "user_id"
                fig = px.bar(
                    summary,
                    x=label_col,
                    y="avg_productivity",
                    color="role",
                    text="avg_productivity",
                    title="Average Productivity by User",
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(xaxis_title="User", yaxis_title="Avg Productivity", xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

                # --- Team Productivity Status ---
                st.subheader("🔎 Team Productivity Status")
                fig_status = px.pie(
                    summary,
                    names="status",
                    title="Team Productivity Split",
                    hole=0.4,
                )
                st.plotly_chart(fig_status, use_container_width=True)

                # --- Role Distribution ---
                st.subheader("👥 Role Distribution")
                if "role" in df_log.columns:
                    fig_roles = px.bar(
                        df_log["role"].value_counts().reset_index(),
                        x="index",
                        y="role",
                        text="role",
                        title="Role Distribution",
                    )
                    fig_roles.update_traces(texttemplate="%{text}", textposition="outside")
                    fig_roles.update_layout(xaxis_title="Role", yaxis_title="Count")
                    st.plotly_chart(fig_roles, use_container_width=True)

                # --- Drilldown ---
                st.subheader("🔍 Inspect a User (drilldown)")
                if "username" in df_log.columns:
                    user_choice = st.selectbox("Select user", sorted(df_log["username"].unique()))
                    user_data = df_log[df_log["username"] == user_choice].sort_values("timestamp")
                else:
                    user_choice = st.selectbox("Select user_id", sorted(df_log["user_id"].unique()))
                    user_data = df_log[df_log["user_id"] == user_choice].sort_values("timestamp")

                if not user_data.empty and "productivity" in user_data.columns:
                    fig_user = px.line(
                        user_data,
                        x="timestamp",
                        y="productivity",
                        title=f"Productivity Trend — {user_choice}",
                    )
                    st.plotly_chart(fig_user, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error in admin dashboard: {e}")

    # --- Option 2: Loaded Dataset ---
    elif admin_option == "Loaded Dataset":
        st.header("📂 Preview Dataset")
        try:
            dataset_choice = st.selectbox("Select dataset", ["Train", "Test"])
            file_path = os.path.join("data", "train.csv" if dataset_choice == "Train" else "test.csv")

            if os.path.exists(file_path):
                df_data = pd.read_csv(file_path)
                st.success(f"Showing first 50 rows of {dataset_choice} dataset:")
                st.dataframe(df_data.head(50), use_container_width=True)
                st.write("Shape:", df_data.shape)
            else:
                st.error(f"Dataset file not found: {file_path}")

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

    # --- Option 3: Self Productivity Testing ---
    elif admin_option == "Self Productivity Testing":
        st.header("🧪 Admin Self Productivity Test")
        st.write("This runs the realtime pipeline but does NOT save into logs (for demo only).")

        st_autorefresh(interval=5000, key="admin_realtime_refresh")

        try:
            reg_model = joblib.load("models/regression_model.pkl")
            cluster_model = joblib.load("models/clustering_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            train_feature_list = joblib.load("models/train_feature_list.pkl")
            cluster_role_map = joblib.load("models/cluster_role_map.pkl")

            latest_row = get_realtime_data()
            for col in train_feature_list:
                if col not in latest_row.columns:
                    latest_row[col] = 0
            X = latest_row[train_feature_list].astype(float)
            X_scaled = scaler.transform(X)

            predicted_productivity = round(float(reg_model.predict(X_scaled)[0]), 2)
            predicted_cluster = int(cluster_model.predict(X_scaled)[0])
            predicted_role = cluster_role_map.get(predicted_cluster, f"Cluster {predicted_cluster}")

            status = "✅ Productive" if predicted_productivity >= PRODUCTIVITY_THRESHOLD else "❌ Not Productive"

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Productivity", f"{predicted_productivity:.2f}")
            col2.metric("Predicted Role", predicted_role)
            col3.metric("Status", status)

            st.subheader("Raw Capture")
            st.write(latest_row)

        except Exception as e:
            st.error(f"❌ Error in self productivity test: {e}")

    # ===== Admin: Create User =====
    st.markdown("---")
    st.subheader("🔐 Admin — Create User")
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
                    st.error("⚠️ Username already exists. Please choose another.")

    # ===== Admin: Reset User Password =====
    st.markdown("---")
    st.subheader("🔑 Admin — Reset User Password")
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
                        conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?",
                                     (new_hash, new_salt, selected_user))
                        conn.commit()
                    st.success(f"✅ Password for '{selected_user}' has been reset.")
                    st.experimental_rerun()
        else:
            st.warning("⚠️ No users found in database.")
