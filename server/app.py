# server/app.py
from flask import Flask, request, jsonify
import sqlite3, os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = Flask(__name__)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/log", methods=["POST"])
def add_log():
    """Users push one realtime log here"""
    try:
        data = request.json or {}
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO logs (
                    user_id, username, timestamp,
                    typing_intensity, mouse_intensity, idle_duration,
                    task_switch_delta, task_switch_total,
                    time_coding, time_browsing, time_other,
                    productivity, role, role_group, status
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                data.get("user_id"),
                data.get("username"),
                data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get("typing_intensity", 0),
                data.get("mouse_intensity", 0),
                data.get("idle_duration", 0),
                data.get("task_switch_delta", 0),
                data.get("task_switch_total", 0),
                data.get("time_coding", 0),
                data.get("time_browsing", 0),
                data.get("time_other", 0),
                data.get("productivity", 0),
                data.get("role", ""),
                data.get("role_group", ""),
                data.get("status", "")
            ))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/logs", methods=["GET"])
def get_logs():
    """Admin dashboard fetches logs"""
    try:
        with get_conn() as conn:
            rows = conn.execute("SELECT * FROM logs ORDER BY datetime(timestamp) DESC").fetchall()
        return jsonify([dict(x) for x in rows])
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # run on 0.0.0.0 so other PCs in LAN can reach it
    app.run(host="0.0.0.0", port=5000, debug=False)
