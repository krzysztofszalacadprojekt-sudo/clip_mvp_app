import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environmental overrides if present
load_dotenv()

# =====================================================================
# 1. CORE PROJECT ROUTING (PyInstaller Safe)
# =====================================================================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================================
# 2. DIRECTORY STRUCTURE
# =====================================================================
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", BASE_DIR / "images"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", BASE_DIR / "embeddings"))

# =====================================================================
# 3. GRAPH TARGETS & AUTO-DETECTION HYBRID
# =====================================================================
# Paths for the two possible deployment scenarios
UNIFIED_MODEL_PATH = MODELS_DIR / os.getenv("UNIFIED_MODEL_NAME", "model.onnx")
VISUAL_MODEL_PATH = MODELS_DIR / os.getenv("VISUAL_MODEL_NAME", "clip_visual.onnx")
TEXT_MODEL_PATH = MODELS_DIR / os.getenv("TEXT_MODEL_NAME", "clip_text.onnx")

TOKENIZER_DIR = MODELS_DIR / os.getenv("TOKENIZER_DIR_NAME", "clip_tokenizer")
TOKENIZER_CONFIG_PATH = TOKENIZER_DIR / os.getenv("TOKENIZER_CONFIG_NAME", "config.json")

# --- THE SMART SWITCH ---
# Dynamically checks if a unified graph (like SigLIP 2) is sitting in the folder.
# Your backend initialization will read this flag to pick 1 vs 2 engine sessions.
IS_UNIFIED = UNIFIED_MODEL_PATH.exists()

# --- TOKENS / TEXT LENGTH ---
# Use tokenizer/model metadata when available to avoid shape mismatches.
TEXT_MAX_LENGTH = 64
IMAGE_SIZE = 256
try:
    if TOKENIZER_CONFIG_PATH.exists():
        with open(TOKENIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
            tokenizer_meta = json.load(f)
        text_config = tokenizer_meta.get("text_config", {})
        max_pos = text_config.get("max_position_embeddings")
        if isinstance(max_pos, int) and 1 <= max_pos <= 512:
            TEXT_MAX_LENGTH = max_pos

        vision_config = tokenizer_meta.get("vision_config", {})
        image_size = vision_config.get("image_size")
        if isinstance(image_size, int) and image_size > 0:
            IMAGE_SIZE = image_size
except Exception:
    TEXT_MAX_LENGTH = int(os.getenv("TEXT_MAX_LENGTH", TEXT_MAX_LENGTH))
    if TEXT_MAX_LENGTH <= 0 or TEXT_MAX_LENGTH > 512:
        TEXT_MAX_LENGTH = 64

    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", IMAGE_SIZE))
    if IMAGE_SIZE <= 0:
        IMAGE_SIZE = 256

# =====================================================================
# 4. VECTOR DATABASE ASSETS
# =====================================================================
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
IMAGE_PATHS_LIST = EMBEDDINGS_DIR / "image_paths.json"


# =====================================================================
# 5. ENVIRONMENT SELF-HEALING
# =====================================================================
for directory in [MODELS_DIR, IMAGES_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)