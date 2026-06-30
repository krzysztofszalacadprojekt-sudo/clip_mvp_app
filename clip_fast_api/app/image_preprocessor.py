from PIL import Image
from transformers import AutoProcessor  # 1. Switched to generic AutoProcessor
from .config import MODELS_DIR, TOKENIZER_DIR

# Lazy initialization global
processor = None


def _init_processor():
    global processor
    if processor is None:
        print("Initializing model processor...")
        
        if not MODELS_DIR.exists():
            raise FileNotFoundError(f"Processor directory not found at: {MODELS_DIR}")
            
        # AutoProcessor dynamically reads the unique preprocessor_config.json
        # ✅ POPRAWIONY, CZYSTY KOD:
        processor = AutoProcessor.from_pretrained(
            str(TOKENIZER_DIR), 
            local_files_only=True
        )
        print("Processor initialized.")

def preprocess_image(image_path: str):
    """
    Loads and preprocesses an image dynamically matching the active model's required shape.
    """
    _init_processor()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]
    return pixel_values