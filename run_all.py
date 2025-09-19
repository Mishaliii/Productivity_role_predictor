# run_all.py
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")

processes = []

try:
    # Run app.py on port 8501
    processes.append(subprocess.Popen(["streamlit", "run", os.path.join(APP_DIR, "app.py"), "--server.port=8501"]))
    # Run dashboard.py on port 8502
    processes.append(subprocess.Popen(["streamlit", "run", os.path.join(APP_DIR, "dashboard.py"), "--server.port=8502"]))

    print("✅ Both apps running:")
    print("   👉 User App: http://localhost:8501")
    print("   👉 Dashboard: http://localhost:8502")

    # Wait for them
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("🛑 Stopping all apps...")
    for p in processes:
        p.terminate()
