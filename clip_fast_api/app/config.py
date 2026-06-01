from pathlib import Path
import sys

# Paths
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent.parent

MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# ONNX Models
VISUAL_MODEL_PATH = MODELS_DIR / "clip_visual_vitb32.onnx"
TEXT_MODEL_PATH = MODELS_DIR / "clip_text_vitb32.onnx"
TOKENIZER_DIR = MODELS_DIR / "clip_tokenizer"

# FAISS Index
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
IMAGE_PATHS_LIST = EMBEDDINGS_DIR / "image_paths.json"

# Create embeddings directory if it doesn't exist
EMBEDDINGS_DIR.mkdir(exist_ok=True)
