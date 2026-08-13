import os
import sys
import subprocess

def main():
    print("==================================================")
    print("🚀 Starting iCloudEMS Campus Intelligence API Server...")
    print("   Decoupled MJPEG + WebSocket Glassmorphic Dashboard")
    print("==================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn_bin = os.path.join(script_dir, "venv", "bin", "uvicorn")
    
    if not os.path.exists(uvicorn_bin):
        print("Virtual environment uvicorn binary not found!")
        sys.exit(1)
        
    cmd = [uvicorn_bin, "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")

if __name__ == "__main__":
    main()
