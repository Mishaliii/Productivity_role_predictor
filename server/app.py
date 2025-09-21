# server/app.py
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# allow importing project modules (script/db_manager.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from script.db_manager import init_db, insert_log, get_conn
import pandas as pd

app = Flask(__name__)
CORS(app)

# Ensure DB exists (this uses the same DB file from script/db_manager.py)
init_db()

@app.route("/")
def home():
    return jsonify({"message": "Flask server is running"})

@app.route("/api/logs", methods=["POST"])
def receive_log():
    """
    Accept POSTed JSON logs from user Streamlit app.
    Acceptable payloads:
      1) { "user_id": 2, "Timestamp": "...", "PredictedProductivity": 70, ... }
      2) { "user_id": 2, "row": { "Timestamp": "...", ... } }
    The endpoint will insert into the same db via insert_log(user_id, row_dict).
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "empty payload"}), 400

        user_id = data.get("user_id")
        if user_id is None:
            return jsonify({"error": "user_id is required in payload"}), 400

        # Accept either `row` dict or top-level fields
        row = data.get("row")
        if row is None:
            # create row from entire payload minus user_id
            row = {k: v for k, v in data.items() if k != "user_id"}

        # ensure Timestamp exists
        if "Timestamp" not in row or not row.get("Timestamp"):
            row["Timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Insert using existing db manager (this will save to your project's db)
        insert_log(int(user_id), row)

        return jsonify({"ok": True}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/logs", methods=["GET"])
def fetch_logs():
    """
    Return logs. Optional query param: user_id
    Example: GET /api/logs?user_id=2
    """
    try:
        user_id = request.args.get("user_id")
        if user_id:
            df = pd.DataFrame()
            try:
                df = pd.read_json(get_logs(user_id=int(user_id)).to_json(orient="records"))
            except Exception:
                df = pd.DataFrame()
        else:
            df = get_logs()
        # if df is a pandas DataFrame (as get_logs returns), convert to records
        if hasattr(df, "to_json"):
            return df.to_json(orient="records"), 200
        else:
            return jsonify([]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") in ("1", "true", "True")
    print(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
