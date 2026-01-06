import os
import subprocess
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
