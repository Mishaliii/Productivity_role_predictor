# script/db_manager.py
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime
import hashlib, binascii
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "database.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# PBKDF2 settings (no external lib needed)
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

def _hash_password(password: str):
    salt = os.urandom(SALT_BYTES)
    pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(pwdhash).decode(), binascii.hexlify(salt).decode()

def _verify_password(stored_hash_hex: str, stored_salt_hex: str, provided_password: str) -> bool:
    salt = binascii.unhexlify(stored_salt_hex)
    pwdhash = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(pwdhash).decode() == stored_hash_hex

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        # Users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin','user')) NOT NULL DEFAULT 'user'
        );
        """)
        # Logs table with username included
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

def add_user(username: str, password: str, role: str = "user") -> bool:
    """Return True if added, False if username exists."""
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
    """Returns (user_id, role) if OK, otherwise None"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id,password_hash,salt,role FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        if not row:
            return None
        if _verify_password(row["password_hash"], row["salt"], password):
            return (row["user_id"], row["role"])
        return None

def change_password(username: str, old_password: str, new_password: str) -> bool:
    auth = authenticate_user(username, old_password)
    if not auth:
        return False
    new_hash, new_salt = _hash_password(new_password)
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?", (new_hash, new_salt, username))
        conn.commit()
    return True

def list_users():
    with get_conn() as conn:
        return pd.read_sql_query("SELECT user_id, username, role FROM users", conn)

def insert_log(user_id: int, row: dict):
    ts = row.get("Timestamp") or datetime.utcnow().isoformat()

    # fetch username for this user_id
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE user_id=?", (user_id,))
        result = cur.fetchone()
        username = result["username"] if result else f"user_{user_id}"

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO logs (
                user_id, username, timestamp, typing_intensity, mouse_intensity, idle_duration,
                task_switch_delta, task_switch_total, time_coding, time_browsing, time_other,
                productivity, role, role_group, status
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            user_id,
            username,
            ts,
            row.get("typing_intensity", 0),
            row.get("mouse_intensity", 0),
            row.get("idle_duration", 0),
            row.get("task_switch_delta", 0),
            row.get("task_switch_total", 0),
            row.get("time_coding", 0),
            row.get("time_browsing", 0),
            row.get("time_other", 0),
            row.get("PredictedProductivity", row.get("productivity", None)),
            row.get("RoleName", row.get("role", "")),
            row.get("RoleGroup", ""),
            row.get("ProductivityStatus", "")
        ))
        conn.commit()

def get_logs(user_id: int = None, limit: int = None) -> pd.DataFrame:
    with get_conn() as conn:
        query = "SELECT * FROM logs"
        params = []
        if user_id is not None:
            query += " WHERE user_id=?"
            params.append(user_id)
        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        df = pd.read_sql_query(query, conn, params=params if params else None)
    return df
