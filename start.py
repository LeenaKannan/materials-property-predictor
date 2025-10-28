import subprocess
import sys
import time
import os

# Check models directory exists
if not os.path.exists("models"):
    print("⚠️  'models' directory not found")
    print("   Create a 'models' folder and put your trained models there:")
    sys.exit(1)

# Check if there are any model files
model_files = [f for f in os.listdir("models") if f.endswith('.pt')]
if not model_files:
    print("⚠️  No .pt files found in models/")
    print("   Copy your trained models to the models/ folder\n")
    sys.exit(1)

print("Starting services...")

# Start API
api = subprocess.Popen([sys.executable, "backend/api/main.py"])
print("✓ API starting...")
time.sleep(3)

# Start Streamlit
ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
print("✓ UI starting...")

print("📍 Web Interface: http://localhost:8501")
print("📍 API: http://localhost:8000")
print("\n Press Ctrl+C to stop\n")

try:
    api.wait()
except KeyboardInterrupt:
    print("\n\n🛑 Stopping...")
    api.terminate()
    ui.terminate()
    time.sleep(1)
    print("✓ Stopped\n")
