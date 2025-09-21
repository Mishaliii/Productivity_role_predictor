# script/collect_realtime_data.py
import pandas as pd
import time
from pynput import keyboard, mouse
import pygetwindow as gw
import math
import socket
import uuid
import os
from datetime import datetime

# ================== GLOBAL VARIABLES ==================
keypress_count = 0
mouse_clicks = 0
mouse_distance = 0.0
last_mouse_position = None
last_activity_time = time.time()
last_fetch_time = time.time()

# Task switching
last_active_title = None
switch_count_total = 0

# Time tracking (cumulative seconds)
time_counters = {
    "time_coding": 0.0,
    "time_browsing": 0.0,
    "time_other": 0.0   # grouped: docs, email, meetings, socialmedia
}

# ================== DEVICE ID (Persistent) ==================
DEVICE_ID_FILE = os.path.join(os.path.dirname(__file__), "..", ".device_id")

def get_device_id():
    """Return a persistent unique ID for this device."""
    if os.path.exists(DEVICE_ID_FILE):
        with open(DEVICE_ID_FILE, "r") as f:
            return f.read().strip()
    # Generate new unique ID
    new_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    with open(DEVICE_ID_FILE, "w") as f:
        f.write(new_id)
    return new_id

DEVICE_ID = get_device_id()

# ================== SESSION ID (Daily Persistent) ==================
def get_session_id():
    """Return a session ID that persists for the day."""
    today = datetime.now().strftime("%Y%m%d")
    return int(today)

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
        mouse_distance += math.sqrt(dx * dx + dy * dy)
    last_mouse_position = (x, y)
    last_activity_time = time.time()

# ================== START LISTENERS (ONCE) ==================
keyboard.Listener(on_press=on_keypress, daemon=True).start()
mouse.Listener(on_click=on_click, on_move=on_move, daemon=True).start()

# ================== ACTIVE WINDOW HELPER ==================
def get_active_window_title():
    try:
        window = gw.getActiveWindow()
        if window is not None:
            return (window.title or "").strip().lower()
    except Exception:
        return ""
    return ""

# ================== FORMATTER ==================
def format_time(seconds: float) -> str:
    """Convert raw seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    else:
        hrs, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs}h {mins}m"

# ================== MAIN FUNCTION ==================
def get_realtime_data():
    global keypress_count, mouse_clicks, mouse_distance
    global last_activity_time, last_fetch_time
    global last_active_title, switch_count_total, time_counters

    now = time.time()
    interval = max(now - last_fetch_time, 1.0)
    last_fetch_time = now

    # Idle time
    idle_time = now - last_activity_time

    # Intensities
    typing_intensity = keypress_count / interval
    mouse_intensity = (mouse_clicks + (mouse_distance / 100.0)) / interval

    # Active window
    active_title = get_active_window_title()

    # ================== Task Switching ==================
    task_switch_delta = 0
    if last_active_title is None:
        last_active_title = active_title
    elif active_title and active_title != last_active_title:
        switch_count_total += 1
        task_switch_delta += 1   # accumulate count properly
        last_active_title = active_title

    # Categories
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
        if any(s in active_title for s in social_sites):
            active_category = "time_other"
        else:
            active_category = "time_browsing"
    elif any(a in active_title for a in meeting_apps + docs_apps + mail_apps):
        active_category = "time_other"

    # Accumulate time (seconds)
    if active_category:
        time_counters[active_category] += interval

    # Data row
    data = {
        "session_id": [SESSION_ID],
        "device_id": [DEVICE_ID],
        "typing_intensity": [round(typing_intensity, 2)],
        "mouse_intensity": [round(mouse_intensity, 2)],
        "idle_duration": [round(idle_time, 2)],
        "task_switch_delta": [task_switch_delta],
        "task_switch_total": [switch_count_total],
        "time_coding": [round(time_counters["time_coding"], 2)],
        "time_browsing": [round(time_counters["time_browsing"], 2)],
        "time_other": [round(time_counters["time_other"], 2)],
        # Human-readable
        "time_coding_str": [format_time(time_counters["time_coding"])],
        "time_browsing_str": [format_time(time_counters["time_browsing"])],
        "time_other_str": [format_time(time_counters["time_other"])],
        "active_window_title": [active_title],
    }

    # Reset interval counters
    keypress_count = 0
    mouse_clicks = 0
    mouse_distance = 0.0

    print(f"DEBUG → Active: {active_title}, ΔSwitch: {task_switch_delta}, TotalSwitch: {switch_count_total}, Cat: {active_category}, TimeCounters: {time_counters}")

    return pd.DataFrame(data)   # ✅ properly indented inside function
