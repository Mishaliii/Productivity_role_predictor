import pandas as pd
import time
from pynput import keyboard, mouse
import pygetwindow as gw
import math
import socket

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
    "time_docs": 0.0,
    "time_email": 0.0,
    "time_meetings": 0.0,
    "time_socialmedia": 0.0,
    "time_browsing": 0.0
}

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

# ================== MAIN FUNCTION ==================
def get_realtime_data():
    """
    Collect activity metrics per interval:
    - typing_intensity, mouse_intensity
    - idle_duration (seconds)
    - task_switch_delta (switches this interval)
    - task_switch_total (cumulative switches)
    - cumulative time_xxx
    """
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

    # Task switching
    task_switch_delta = 0
    if last_active_title is None:
        last_active_title = active_title
    elif active_title and active_title != last_active_title:
        switch_count_total += 1
        task_switch_delta = 1
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
    elif any(a in active_title for a in meeting_apps):
        active_category = "time_meetings"
    elif any(a in active_title for a in docs_apps):
        active_category = "time_docs"
    elif any(a in active_title for a in mail_apps):
        active_category = "time_email"
    elif any(b in active_title for b in browser_apps):
        if any(s in active_title for s in social_sites):
            active_category = "time_socialmedia"
        else:
            active_category = "time_browsing"

    # Accumulate time (seconds)
    if active_category:
        time_counters[active_category] += interval

    # Device ID
    device_id = socket.gethostname()

    # Data row
    data = {
        "session_id": [int(now)],
        "device_id": [device_id],
        "typing_intensity": [round(typing_intensity, 2)],
        "mouse_intensity": [round(mouse_intensity, 2)],
        "idle_duration": [round(idle_time, 2)],  # seconds
        "task_switch_delta": [task_switch_delta],  # new this interval
        "task_switch_total": [switch_count_total],  # cumulative
        "time_coding": [round(time_counters["time_coding"], 2)],
        "time_docs": [round(time_counters["time_docs"], 2)],
        "time_email": [round(time_counters["time_email"], 2)],
        "time_meetings": [round(time_counters["time_meetings"], 2)],
        "time_socialmedia": [round(time_counters["time_socialmedia"], 2)],
        "time_browsing": [round(time_counters["time_browsing"], 2)],
        "active_window_title": [active_title],  # for debug
    }

    # Reset interval counters
    keypress_count = 0
    mouse_clicks = 0
    mouse_distance = 0.0

    print(f"DEBUG → Active: {active_title}, ΔSwitch: {task_switch_delta}, TotalSwitch: {switch_count_total}, Cat: {active_category}, TimeCounters: {time_counters}")

    return pd.DataFrame(data)
