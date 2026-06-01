import sys
from pathlib import Path
import onnxruntime as ort
import numpy as np
from .config import VISUAL_MODEL_PATH, TEXT_MODEL_PATH

# Lazy initialization globals
visual_session = None
text_session = None

def get_safe_model_path(model_path) -> Path:
    """Ensures models are loaded correctly when running as a PyInstaller executable."""
    if getattr(sys, "frozen", False):
        # If compiled, look in the 'models' folder next to the .exe
        return Path(sys.executable).parent / "models" / Path(model_path).name
    return Path(model_path)

def _init_visual_session():
    global visual_session
    if visual_session is None:
        print("Initializing ONNX visual session from local file...")
        safe_path = get_safe_model_path(VISUAL_MODEL_PATH)
        available_providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("Using DirectML (GPU) for visual model.")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️ Using CPU for visual model. (DirectML not available)")
        visual_session = ort.InferenceSession(str(safe_path), providers=providers)
        print(f"Visual session providers: {visual_session.get_providers()}")
        print("Visual session initialized.")


def _init_text_session():
    global text_session
    if text_session is None:
        print("Initializing ONNX text session from local file...")
        safe_path = get_safe_model_path(TEXT_MODEL_PATH)
        available_providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            print("Using DirectML (GPU) for text model.")
        else:
            providers = ['CPUExecutionProvider']
            print("⚠️ Using CPU for text model. (DirectML not available)")
        text_session = ort.InferenceSession(str(safe_path), providers=providers)
        print(f"Text session providers: {text_session.get_providers()}")
        print("Text session initialized.")


def get_image_embedding(pixel_values):
    """
    Generates an embedding for the given image pixel values.
    """
    _init_visual_session()
    input_name = visual_session.get_inputs()[0].name
    output_name = visual_session.get_outputs()[0].name
    
    result = visual_session.run([output_name], {input_name: pixel_values})
    return result[0]

def get_image_embeddings_batch(pixel_values_batch):
    """
    Generates embeddings for a batch of image pixel values.
    """
    _init_visual_session()
    input_name = visual_session.get_inputs()[0].name
    output_name = visual_session.get_outputs()[0].name
    
    result = visual_session.run([output_name], {input_name: pixel_values_batch})
    return result[0]

def get_text_embedding(inputs):
    """
    Generates an embedding for the given text inputs dictionary.
    """
    _init_text_session()
    
    onnx_inputs = {}
    for inp in text_session.get_inputs():
        if inp.name in inputs:
            input_data = inputs[inp.name]
            if 'int32' in inp.type:
                input_data = input_data.astype(np.int32)
            elif 'int64' in inp.type:
                input_data = input_data.astype(np.int64)
            onnx_inputs[inp.name] = input_data
    
    outputs = text_session.run(None, onnx_inputs)
    output_names = [out.name for out in text_session.get_outputs()]
    
    text_embedding = outputs[output_names.index("text_embeds")] if "text_embeds" in output_names else next((out for out in outputs if len(out.shape) == 2), outputs[0])
    return text_embedding
