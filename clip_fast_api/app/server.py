import os
import time
from pathlib import Path
from typing import List

import uvicorn
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoProcessor

from dotenv import load_dotenv

from .image_preprocessor import preprocess_image
from .clip_model import get_text_embedding as get_clip_text_embedding, get_image_embedding
from app import config
from .embedding_store import (
    create_and_save_embeddings,
    load_index_and_paths,
    search_similar_images,
    update_embeddings,
    update_embeddings_from_db,
)
from .config import MODELS_DIR, EMBEDDINGS_DIR, DB_PATH

# Load environment variables from .env file
load_dotenv()

# Set HF Token if needed for the tokenizer loading logic
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("⚠️  HF_TOKEN environment variable not set. This may be required for some models.")

import traceback
from contextlib import asynccontextmanager

# --- SCHEMAS ---

class TextPrompt(BaseModel):
    text: str
    top_k: int = 5

class UpdateDirectories(BaseModel):
    directories: List[str]

# --- LIFECYCLE EVENTS ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, image_paths

    print("🚀 Server startup initiated...")

    Path("temp").mkdir(exist_ok=True)

    try:
        # ---------------------------
        # TOKENIZER
        # ---------------------------
        print("STEP 1: Loading tokenizer...")
        start = time.time()

        if not config.TOKENIZER_DIR.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {config.TOKENIZER_DIR}")

        app.state.tokenizer = AutoTokenizer.from_pretrained(
            str(config.TOKENIZER_DIR),
            local_files_only=True,
            fix_mistral_regex=False
        )

        print(f"✅ Tokenizer loaded in {time.time() - start:.2f}s")

        # ---------------------------
        # FAISS
        # ---------------------------
        print("STEP 2: Loading FAISS index...")

        try:
            index, image_paths = load_index_and_paths()

            if index is not None:
                print("✅ FAISS Index loaded successfully.")
            else:
                print("⚠️ No existing index found.")

        except Exception as e:
            print(f"❌ Failed loading FAISS index: {e}")
            traceback.print_exc()

            index = None
            image_paths = []

        # ---------------------------
        # STEP 3: BACKGROUND AUTOMATIC SYNC
        # ---------------------------
        from threading import Thread
        from .embedding_store import update_embeddings_from_db  # Upewnij się, że ścieżka importu pasuje

        def auto_sync_worker():
            global index, image_paths
            try:
                print("🔍 [Background Sync] Checking SQLite for new models missing embeddings...")
                sync_start = time.time()
                
                # Przetwarzanie bazy danych SigLIP-em w osobnym wątku
                message = update_embeddings_from_db(DB_PATH)
                print(f"ℹ️ [Background Sync] {message}")
                
                # Natychmiastowe załadowanie świeżo wygenerowanych wektorów do działającego RAM-u FastAPI
                index, image_paths = load_index_and_paths()
                print(f"🔄 [Background Sync] RAM Hot-reload complete in {time.time() - sync_start:.2f}s.")
            except Exception as sync_err:
                print(f"❌ [Background Sync Error] Auto-update failed: {sync_err}")

        # Odpalamy pracownika w tle. daemon=True gwarantuje, że wątek zamknie się, gdy wyłączysz serwer.
        sync_thread = Thread(target=auto_sync_worker, daemon=True)
        sync_thread.start()

    except Exception as e:
        print(f"❌ FATAL INIT ERROR: {e}")
        traceback.print_exc()

        app.state.tokenizer = None

    yield

    print("🛑 Server shutdown.")

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)
index, image_paths = None, None  # Global store for FAISS/Paths


# --- EMBEDDING ENDPOINTS ---

@app.post("/get-embedding")
def get_embedding_endpoint(prompt: TextPrompt, request: Request):
    if not hasattr(request.app.state, 'tokenizer') or not request.app.state.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded.")

    tokenizer = request.app.state.tokenizer

    max_length = config.TEXT_MAX_LENGTH
    configured_max = tokenizer.init_kwargs.get("model_max_length")
    if isinstance(configured_max, int) and 1 <= configured_max <= 512:
        max_length = configured_max

    inputs = tokenizer(
        prompt.text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    try:
        text_embedding = get_clip_text_embedding(inputs)
        return {"embedding": text_embedding[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-embeddings")
def create_embeddings_endpoint():
    start_time = time.time()
    try:
        create_and_save_embeddings()
        global index, image_paths
        index, image_paths = load_index_and_paths()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Create embeddings completed in {elapsed:.2f} seconds.")
        return JSONResponse(content={"message": f"Embeddings created and saved successfully in {elapsed:.2f}s."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-embeddings")
def update_embeddings_endpoint():
    print("📥 Odebrano żądanie aktualizacji embeddingów z bazy danych SQLite.")
    start_time = time.time()
    try:
        
        # 2. Wywołujemy nową funkcję synchronizującą bazę z SigLIP 2 i FAISS
        message = update_embeddings_from_db(DB_PATH)
        
        # 3. Krytyczne: Przeładowujemy indeks w pamięci RAM serwera FastAPI
        global index, image_paths
        index, image_paths = load_index_and_paths()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Update embeddings completed in {elapsed:.2f} seconds.")
        
        return JSONResponse(content={"message": f"{message} (Took {elapsed:.2f}s)"})
        
    except Exception as e:
        print(f"❌ Błąd podczas operacji update-embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- SEARCH ENDPOINTS ---

@app.post("/find-similar-images-by-text")
def find_similar_images_by_text(prompt: TextPrompt, request: Request):
    global index, image_paths
    print(f"Received text search request: {prompt.text} (top_k={prompt.top_k})")
    start_time = time.time()
    
    if index is None or image_paths is None:
        index, image_paths = load_index_and_paths()
        if index is None:
            raise HTTPException(status_code=400, detail="Index not found.")
    
    try:
        active_tokenizer = request.app.state.tokenizer

        max_length = config.TEXT_MAX_LENGTH
        configured_max = active_tokenizer.init_kwargs.get("model_max_length")
        if isinstance(configured_max, int) and 1 <= configured_max <= 512:
            max_length = configured_max

        # Przygotowanie danych pod ONNX Runtime
        inputs = active_tokenizer(
            prompt.text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        
        text_embedding = get_clip_text_embedding(inputs)
        
        results = search_similar_images(text_embedding, index, image_paths, prompt.top_k)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Text search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"❌ Search failed: {e}") # Dobrze dodać printa w konsoli serwera do podglądu
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-images-by-image")
async def find_similar_images_by_image(file: UploadFile = File(...), top_k: int = 5):
    global index, image_paths
    print(f"Received image search request (top_k={top_k})")
    start_time = time.time()
    if index is None:
        raise HTTPException(status_code=400, detail="Index not loaded.")

    try:
        temp_path = None
        try:
            temp_path = Path("temp") / file.filename
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())

            pixel_values = preprocess_image(str(temp_path))
            image_embedding = get_image_embedding(pixel_values)
            
            results = search_similar_images(image_embedding, index, image_paths, top_k)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

        elapsed = time.time() - start_time
        print(f"⏱️ Image search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def root():
    return {"status": "server is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# --- MAIN RUNNER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)