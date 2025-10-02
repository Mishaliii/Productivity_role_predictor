import os
import sqlite3
import hashlib
import binascii
import requests
from contextlib import contextmanager
from datetime import datetime
import pandas as pd

# ===================== CONFIG ===================== #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")
print("DEBUG → DB_PATH used:", DB_PATH)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 👉 Set this to the ADMIN machine IP (where Flask server runs)
SERVER_URL = "http://192.168.220.34:5000"   # change if admin IP changes

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ================= PASSWORD UTILS ================= #
def _hash_password(password: str):
    salt = os.urandom(SALT_BYTES)
    pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(pwdhash).decode(), binascii.hexlify(salt).decode()


def _verify_password(stored_hash_hex: str, stored_salt_hex: str, provided_password: str) -> bool:
    salt = binascii.unhexlify(stored_salt_hex)
    pwdhash = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(pwdhash).decode() == stored_hash_hex


# ================= INIT DB ================= #
def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin','user')) NOT NULL DEFAULT 'user'
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            typing_intensity REAL,
            mouse_intensity REAL,
            idle_duration REAL,
            task_switch_delta INTEGER,
            task_switch_total INTEGER,
            time_coding REAL,
            time_browsing REAL,
            time_other REAL,
            productivity REAL,
            role TEXT,
            role_group TEXT,
            status TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        """)
        conn.commit()


# ================= USER OPS ================= #
def add_user(username: str, password: str, role: str = "user") -> bool:
    pwd_hash, salt = _hash_password(password)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username,password_hash,salt,role) VALUES (?,?,?,?)",
                (username, pwd_hash, salt, role)
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id,password_hash,salt,role FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return None
        if _verify_password(row["password_hash"], row["salt"], password):
            return (row["user_id"], row["role"])
        return None


# ================= LOG OPS ================= #
def insert_log(user_id: int, row: dict):
    """
    Insert one realtime activity record into local DB
    AND also push it to the central admin server if available.
    """
    ts = row.get("Timestamp") or row.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "user_id": user_id,
        "username": row.get("username"),
        "timestamp": ts,
        "typing_intensity": row.get("typing_intensity", 0),
        "mouse_intensity": row.get("mouse_intensity", 0),
        "idle_duration": row.get("idle_duration", 0),
        "task_switch_delta": row.get("task_switch_delta", 0),
        "task_switch_total": row.get("task_switch_total", 0),
        "time_coding": row.get("time_coding", 0),
        "time_browsing": row.get("time_browsing", 0),
        "time_other": row.get("time_other", 0),
        "productivity": row.get("PredictedProductivity", row.get("productivity")),
        "role": row.get("RoleName", row.get("role")),
        "role_group": row.get("RoleGroup", ""),
        "status": row.get("ProductivityStatus", row.get("status", "")),
    }

    # ---- Local DB insert ----
    try:
        with get_conn() as conn:
            username_db = conn.execute(
                "SELECT username FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
            if username_db and username_db["username"]:
                record["username"] = username_db["username"]

            conn.execute(
                """
                INSERT INTO logs (
                    user_id, username, timestamp,
                    typing_intensity, mouse_intensity, idle_duration,
                    task_switch_delta, task_switch_total,
                    time_coding, time_browsing, time_other,
                    productivity, role, role_group, status
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record["user_id"], record["username"], record["timestamp"],
                    record["typing_intensity"], record["mouse_intensity"], record["idle_duration"],
                    record["task_switch_delta"], record["task_switch_total"],
                    record["time_coding"], record["time_browsing"], record["time_other"],
                    record["productivity"], record["role"], record["role_group"], record["status"]
                )
            )
            conn.commit()
    except Exception as e:
        print(f"[DB insert_log ERROR] {e}")

    # ---- Push to central server ----
    try:
        resp = requests.post(f"{SERVER_URL}/log", json=record, timeout=2)
        if resp.status_code != 200:
            print(f"[insert_log] Push failed {resp.status_code}: {resp.text}")
    except Exception as e:
        print("[insert_log] push failed:", e)


def get_logs(user_id: int = None, limit: int = None) -> pd.DataFrame:
    """Fetch logs (optionally for one user). No silent limit unless passed."""
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


def change_password(username: str, old_password: str, new_password: str) -> bool:
    auth = authenticate_user(username, old_password)
    if not auth:
        return False
    new_hash, new_salt = _hash_password(new_password)
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE username=?",
            (new_hash, new_salt, username)
        )
        conn.commit()
    return True
