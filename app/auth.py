# app/auth.py
import streamlit as st
import os, sys

# Make sure project root is on sys.path so "script" package is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from script.db_manager import init_db, authenticate_user, add_user, change_password

def ensure_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "role" not in st.session_state:
        st.session_state.role = None

def logout():
    for k in ["logged_in","username","user_id","role"]:
        st.session_state.pop(k, None)
    st.rerun()

def show_login():
    """
    Returns True if user is logged in (session_state updated), otherwise False.
    Use in app/app.py like:
        if not show_login(): st.stop()
    """
    ensure_session()
    init_db()

    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in: {st.session_state.username} ({st.session_state.role})")

        # -------- CHANGE PASSWORD (user self-service) --------
        with st.sidebar.expander("🔑 Change Password"):
            old_pass = st.text_input("Old password", type="password", key="cp_old")
            new_pass = st.text_input("New password", type="password", key="cp_new")
            if st.button("Update Password", key="cp_btn"):
                if not old_pass or not new_pass:
                    st.error("❌ Please enter both old and new password.")
                else:
                    ok = change_password(st.session_state.username, old_pass, new_pass)
                    if ok:
                        st.success("✅ Password updated successfully.")
                    else:
                        st.error("❌ Incorrect old password.")

        if st.sidebar.button("Logout"):
            logout()
        return True

    st.sidebar.title("🔐 Account")
    action = st.sidebar.radio("Action", ["Login", "Sign up"])

    # ---------- LOGIN ----------
    if action == "Login":
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            if not username or not password:
                st.sidebar.error("Enter both username and password.")
            else:
                auth = authenticate_user(username, password)
                if auth:
                    st.session_state.user_id, st.session_state.role = auth
                    st.session_state.username = username
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
        return False

    # ---------- SIGN UP ----------
    new_user = st.sidebar.text_input("New username", key="signup_username")
    new_pass = st.sidebar.text_input("New password", type="password", key="signup_password")
    if st.sidebar.button("Create account"):
        if not new_user or not new_pass:
            st.sidebar.error("Please enter both username and password")
        else:
            ok = add_user(new_user, new_pass, role="user")
            if ok:
                st.sidebar.success("Account created. Now login.")
            else:
                st.sidebar.error("Username already exists")
    return False
