import sys
import numpy as np
import onnxruntime as ort
from . import config

shared_session = None  
visual_session = None
text_session = None  

def _build_session(model_path):
    """
    Pomocnicza funkcja centralizująca dobór akceleracji sprzętowej (DirectML / CPU).
    """
    available_providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in available_providers:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        print(f"🚀 DirectML (GPU) aktywowane dla: {model_path.name}")
    else:
        providers = ['CPUExecutionProvider']
        print(f"⚠️ DirectML niedostępne. CPU aktywowane dla: {model_path.name}")
        
    return ort.InferenceSession(str(model_path), providers=providers)


def _get_active_session(session_type: str):
    """
    Zwraca odpowiednią sesję w zależności od struktury plików wykrytej w config.py.
    """
    global shared_session, visual_session, text_session
    
    if config.IS_UNIFIED:
        if shared_session is None:
            shared_session = _build_session(config.UNIFIED_MODEL_PATH)
        return shared_session
    
    if session_type == "visual":
        if visual_session is None:
            visual_session = _build_session(config.VISUAL_MODEL_PATH)
        return visual_session
    elif session_type == "text":
        if text_session is None:
            text_session = _build_session(config.TEXT_MODEL_PATH)
        return text_session

def _run_vision_inference(pixel_data):
    session = _get_active_session("visual")
    onnx_inputs = {}

    batch_size = pixel_data.shape[0]
    default_sequence_length = 64

    for inp in session.get_inputs():
        if "pixel" in inp.name.lower():
            onnx_inputs[inp.name] = pixel_data
        elif config.IS_UNIFIED:
            if 'int32' in inp.type:
                onnx_inputs[inp.name] = np.zeros((batch_size, default_sequence_length), dtype=np.int32)
            elif 'int64' in inp.type:
                onnx_inputs[inp.name] = np.zeros((batch_size, default_sequence_length), dtype=np.int64)

    outputs_meta = session.get_outputs()
    target_output_name = None

    TARGET_VISION_NODE = "image_embeds" 
    
    return session.run([TARGET_VISION_NODE], onnx_inputs)[0]


def get_image_embedding(pixel_values):
    """Generuje embedding dla pojedynczego zdjęcia."""
    return _run_vision_inference(pixel_values)


def get_image_embeddings_batch(pixel_values_batch):
    """Generuje embeddingi dla paczki zdjęć (Batch processing)."""
    return _run_vision_inference(pixel_values_batch)

def get_text_embedding(inputs):
    """
    Generuje embedding dla podanego słownika wejść tokenizera.
    Bezpieczny dla pojedynczych zapytań oraz masowego przetwarzania (Batch).
    """
    session = _get_active_session("text")
    onnx_inputs = {}
    
    # 🚀 POPRAWKA 1: Dynamicznie pobieramy rozmiar paczki z wejść tokenizera
    # Szukamy klucza 'input_ids', który zawsze określa wielkość paczki tekstu
    batch_size = 1
    if "input_ids" in inputs:
        batch_size = inputs["input_ids"].shape[0]
    
    # Mapowanie danych z tokenizera do grafu obliczeniowego
    for inp in session.get_inputs():
        if inp.name in inputs:
            input_data = inputs[inp.name]
            if inp.name == "input_ids" and config.IS_UNIFIED:
                expected_length = config.TEXT_MAX_LENGTH
                if input_data.ndim == 2 and input_data.shape[1] != expected_length:
                    raise ValueError(
                        f"Unified model requires input_ids length {expected_length}, "
                        f"but got {input_data.shape[1]}. Ensure tokenizer max_length={expected_length}."
                    )
            # Bezpieczne rzutowanie typów numerycznych pod silnik ONNX
            if 'int32' in inp.type:
                input_data = input_data.astype(np.int32)
            elif 'int64' in inp.type:
                input_data = input_data.astype(np.int64)
            onnx_inputs[inp.name] = input_data
            
        elif config.IS_UNIFIED and "pixel" in inp.name.lower():
            # 🚀 POPRAWKA 2: Atrapa obrazu idealnie dopasowuje się do rozmiaru paczki tekstów
            resolution = config.IMAGE_SIZE
            onnx_inputs[inp.name] = np.zeros((batch_size, 3, resolution, resolution), dtype=np.float32)

    # 🚀 POPRAWKA 3: Precyzyjne celowanie w wyjście (bez niepotrzebnego run(None))
    TARGET_TEXT_NODE = "text_embeds"
    
    output_names = [out.name for out in session.get_outputs()]
    if TARGET_TEXT_NODE in output_names:
        return session.run([TARGET_TEXT_NODE], onnx_inputs)[0]
    else:
        # Fallback dla starych / nie-zunifikowanych modeli
        fallback_node = next((out.name for out in session.get_outputs() if len(out.shape) == 2), output_names[0])
        return session.run([fallback_node], onnx_inputs)[0]