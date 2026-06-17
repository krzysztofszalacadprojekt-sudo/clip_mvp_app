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
    
    # Mapowanie wejść: Uniwersalna pętla obsługująca nazwy węzłów dla CLIP i SigLIP
    for inp in session.get_inputs():
        if "pixel" in inp.name.lower():
            onnx_inputs[inp.name] = pixel_data
        elif config.IS_UNIFIED:
            # Maskowanie sygnatury tekstu atrapami danych (wymóg zunifikowanego grafu)
            if 'int32' in inp.type:
                onnx_inputs[inp.name] = np.zeros((1, 1), dtype=np.int32)
            elif 'int64' in inp.type:
                onnx_inputs[inp.name] = np.zeros((1, 1), dtype=np.int64)

    output_names = [out.name for out in session.get_outputs()]
    target_output = "image_embeds" if "image_embeds" in output_names else output_names[0]
    
    return session.run([target_output], onnx_inputs)[0]


def get_image_embedding(pixel_values):
    """Generuje embedding dla pojedynczego zdjęcia."""
    return _run_vision_inference(pixel_values)


def get_image_embeddings_batch(pixel_values_batch):
    """Generuje embeddingi dla paczki zdjęć (Batch processing)."""
    return _run_vision_inference(pixel_values_batch)

def get_text_embedding(inputs):
    """
    Generuje embedding dla podanego słownika wejść tokenizera.
    """
    session = _get_active_session("text")
    onnx_inputs = {}
    
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
            # Jeśli model jest zunifikowany, a my liczymy tekst, podajemy puste zdjęcie-atrapę.
            # Use the configured vision image size, not a filename heuristic.
            resolution = config.IMAGE_SIZE
            onnx_inputs[inp.name] = np.zeros((1, 3, resolution, resolution), dtype=np.float32)

    outputs = session.run(None, onnx_inputs)
    output_names = [out.name for out in session.get_outputs()]
    
    # Inteligentne wyciąganie warstwy embeddingu tekstowego
    if "text_embeds" in output_names:
        text_embedding = outputs[output_names.index("text_embeds")]
    else:
        text_embedding = next((out for out in outputs if len(out.shape) == 2), outputs[0])
        
    return text_embedding