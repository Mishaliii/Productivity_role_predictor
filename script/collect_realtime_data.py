# script/collect_realtime_data.py
import pandas as pd
import time
from datetime import datetime
import math
import socket
import uuid
import os

# --- Optional imports (safe fallback if unavailable) ---
try:
    from pynput import keyboard, mouse
except Exception:
    keyboard = None
    mouse = None
try:
    import pygetwindow as gw
except Exception:
    gw = None

# ================== GLOBAL STATE ==================
keypress_count = 0
mouse_clicks = 0
mouse_distance = 0.0
last_mouse_position = None
last_activity_time = time.time()
last_fetch_time = time.time()

# Task switching
last_active_title = ""
switch_count_total = 0

# Time tracking (cumulative seconds)
time_counters = {
    "time_coding": 0.0,
    "time_browsing": 0.0,
    "time_other": 0.0
}

# ================== DEVICE ID ==================
DEVICE_ID_FILE = os.path.join(os.path.dirname(__file__), "..", ".device_id")


def get_device_id() -> str:
    """Return a persistent unique ID for this device."""
    try:
        if os.path.exists(DEVICE_ID_FILE):
            with open(DEVICE_ID_FILE, "r") as f:
                return f.read().strip()
        new_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        with open(DEVICE_ID_FILE, "w") as f:
            f.write(new_id)
        return new_id
    except Exception:
        return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


DEVICE_ID = get_device_id()

# ================== SESSION ID ==================


def get_session_id() -> int:
    """Return a session ID that persists for the day."""
    return int(datetime.now().strftime("%Y%m%d"))


SESSION_ID = get_session_id()

# ================== EVENT HANDLERS ==================


def on_keypress(key):
    global keypress_count, last_activity_time
    keypress_count += 1
    last_activity_time = time.time()


def on_click(x, y, button, pressed):
    global mouse_clicks, last_activity_time
    if pressed:
        mouse_clicks += 1
        last_activity_time = time.time()


def on_move(x, y):
    global mouse_distance, last_mouse_position, last_activity_time
    if last_mouse_position is not None:
        dx = x - last_mouse_position[0]
        dy = y - last_mouse_position[1]
        mouse_distance += math.hypot(dx, dy)
    last_mouse_position = (x, y)
    last_activity_time = time.time()


# Start listeners once (if library available)
if keyboard and mouse:
    try:
        keyboard.Listener(on_press=on_keypress, daemon=True).start()
        mouse.Listener(on_click=on_click, on_move=on_move, daemon=True).start()
    except Exception:
        # Don't crash if listeners fail (e.g., headless env)
        pass

# ================== ACTIVE WINDOW ==================


def get_active_window_title() -> str:
    try:
        if gw:
            window = gw.getActiveWindow()
            if window:
                return (window.title or "").strip().lower()
    except Exception:
        pass
    return ""

# ================== UTIL ==================


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    else:
        hrs, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs}h {mins}m"

# ================== MAIN ==================


def get_realtime_data() -> pd.DataFrame:
    """
    Capture current activity metrics since last call.
    Returns a one-row DataFrame with consistent numeric columns.
    """
    global keypress_count, mouse_clicks, mouse_distance
    global last_activity_time, last_fetch_time
    global last_active_title, switch_count_total, time_counters

    now = time.time()
    interval = max(now - last_fetch_time, 1.0)  # at least 1 sec
    last_fetch_time = now

    # Idle time
    idle_time = now - last_activity_time

    # Event intensities
    typing_intensity = keypress_count / interval
    mouse_intensity = (mouse_clicks + (mouse_distance / 100.0)) / interval

    # Active window
    active_title = get_active_window_title()

    # --- Task Switching ---
    task_switch_delta = 0
    if not last_active_title:
        last_active_title = active_title
    elif active_title and active_title != last_active_title:
        switch_count_total += 1
        task_switch_delta = 1
        last_active_title = active_title

    # --- Categorize Window ---
    coding_apps = ["vscode", "visual studio", "visual studio code", "pycharm",
                   "sublime", "notepad++", "atom", "intellij", "eclipse",
                   "github", "stack overflow", "code"]
    meeting_apps = ["zoom", "microsoft teams", "teams", "meet", "skype", "webex"]
    docs_apps = ["word", "excel", "powerpoint", "notepad", "onenote",
                 "google docs", "libreoffice", "sheet", "slides", "notion", "confluence"]
    mail_apps = ["gmail", "outlook", "mail", "thunderbird", "inbox"]
    browser_apps = ["chrome", "edge", "firefox", "safari", "brave", "opera"]
    social_sites = ["youtube", "facebook", "instagram", "whatsapp", "twitter",
                    "tiktok", "reddit", "linkedin", "telegram"]

    active_category = None
    if any(a in active_title for a in coding_apps):
        active_category = "time_coding"
    elif any(b in active_title for b in browser_apps):
        active_category = "time_other" if any(s in active_title for s in social_sites) else "time_browsing"
    elif any(a in active_title for a in meeting_apps + docs_apps + mail_apps):
        active_category = "time_other"

    if active_category:
        time_counters[active_category] += interval

    # --- Data Row ---
    row = {
        "session_id": SESSION_ID,
        "device_id": DEVICE_ID,
        "typing_intensity": float(round(typing_intensity, 2)),
        "mouse_intensity": float(round(mouse_intensity, 2)),
        "idle_duration": float(round(idle_time, 2)),
        "task_switch_delta": int(task_switch_delta),
        "task_switch_total": int(switch_count_total),
        "time_coding": float(round(time_counters["time_coding"], 2)),
        "time_browsing": float(round(time_counters["time_browsing"], 2)),
        "time_other": float(round(time_counters["time_other"], 2)),
        "time_coding_str": format_time(time_counters["time_coding"]),
        "time_browsing_str": format_time(time_counters["time_browsing"]),
        "time_other_str": format_time(time_counters["time_other"]),
        "active_window_title": active_title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Reset counters for next interval
    keypress_count = 0
    mouse_clicks = 0
    mouse_distance = 0.0

    # Ensure DataFrame with clean dtypes
    df = pd.DataFrame([row])
    numeric_cols = [
        "typing_intensity", "mouse_intensity", "idle_duration",
        "task_switch_delta", "task_switch_total",
        "time_coding", "time_browsing", "time_other"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df
