import uvicorn
import multiprocessing
from app.server import app

if __name__ == "__main__":
    # Required to prevent recursive spawning in compiled Python executables
    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=8000)