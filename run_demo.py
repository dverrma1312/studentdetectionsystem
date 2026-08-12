import os
import sys
import subprocess

def main():
    print("==================================================")
    print("🎥 Starting iCloudEMS Campus Intelligence App...")
    print("==================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, "venv", "bin", "streamlit")
    
    if not os.path.exists(venv_python):
        print("Virtual environment not found! Please wait for package installation.")
        sys.exit(1)
        
    cmd = [venv_python, "run", "app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
