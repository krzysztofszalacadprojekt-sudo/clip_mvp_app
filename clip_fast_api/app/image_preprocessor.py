from PIL import Image
from transformers import CLIPProcessor
from .config import BASE_DIR

# Lazy initialization global
processor = None


def _init_processor():
    global processor
    if processor is None:
        print("Initializing CLIP processor...")
        model_path = BASE_DIR / "client_assets" / "clip_processor"
        
        if not model_path.exists():
            raise FileNotFoundError(f"CLIP processor directory not found at: {model_path}")
            
        processor = CLIPProcessor.from_pretrained(
            str(model_path), 
            local_files_only=True
        )
        print("Processor initialized.")

def preprocess_image(image_path: str):
    """
    Loads and preprocesses an image from the given path.
    """
    _init_processor()
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Ensure consistent size
    inputs = processor(images=image, return_tensors="np")
    return inputs["pixel_values"]
