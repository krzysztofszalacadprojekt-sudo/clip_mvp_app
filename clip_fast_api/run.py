import subprocess
import os
import sys
from pathlib import Path

def run_server():
    """
    Runs the uvicorn server with specified parameters.
    """
    host = "0.0.0.0"
    port = 8000
    workers = int(os.cpu_count() * 0.75) # Use 75% of available cores

    # Detect virtual environment Python
    script_dir = Path(__file__).parent
    venv_python = script_dir / "clip_env" / "Scripts" / "python.exe"
    
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    if python_exe == sys.executable and not Path(sys.executable).name.startswith("clip_env"):
        print(f"WARNING: Using system Python ({python_exe}). Consider activating the virtual environment first.")
    
    # Set environment variable to suppress PyTorch warning
    env = os.environ.copy()
    env["TRANSFORMERS_NO_PYTORCH"] = "1"

    command = [
        python_exe,
        "-m",
        "uvicorn",
        "app.server:app",
        "--host", host,
        "--port", str(port),
        "--reload" # Reloads the server on code changes, useful for development
    ]

    print(f"Starting server with command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Server exited with an error: {e}")
    except KeyboardInterrupt:
        print("Server stopped by user.")

if __name__ == "__main__":
    run_server()
