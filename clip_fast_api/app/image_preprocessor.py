from PIL import Image
from transformers import AutoProcessor  # 1. Switched to generic AutoProcessor
from .config import BASE_DIR, MODELS_DIR, TOKENIZER_DIR

# Lazy initialization global
processor = None


def _init_processor():
    global processor
    if processor is None:
        print("Initializing model processor...")
        
        if not MODELS_DIR.exists():
            raise FileNotFoundError(f"Processor directory not found at: {MODELS_DIR}")
            
        # AutoProcessor dynamically reads the unique preprocessor_config.json
        processor = AutoProcessor.from_pretrained(
            str(TOKENIZER_DIR), 
            local_files_only=True
        )
        print("Processor initialized.")

# def preprocess_image(image_path: str):
#     """
#     Loads and preprocesses an image dynamically matching the active model's required shape.
#     """
#     _init_processor()
#     image = Image.open(image_path).convert("RGB")
    
#     # 3. REMOVED hardcoded image.resize((224, 224)) line entirely.
#     # The processor reads the JSON configuration and handles resizing natively.
#     inputs = processor(images=image, return_tensors="np")
    
#     return inputs["pixel_values"]

def preprocess_image(image_path: str):
    """
    Loads and preprocesses an image dynamically matching the active model's required shape.
    """
    _init_processor()
    image = Image.open(image_path).convert("RGB")
    
    # Check: what size is the raw image?
    print(f"[preprocess_image] Raw image size: {image.size}")
    
    # Processor should resize to 256x256 per config
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]
    
    # Check: what size came out?
    print(f"[preprocess_image] Preprocessed pixel_values shape: {pixel_values.shape}")
    # Should be (1, 3, 256, 256) or (1, 768, 256, 256) depending on order
    
    return pixel_values