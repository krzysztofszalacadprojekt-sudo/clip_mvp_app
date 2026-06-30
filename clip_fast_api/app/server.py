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
    load_index_and_paths,
    search_similar_images,
    update_embeddings_from_db,
    rebuild_index_from_db,
)
from .config import CPP_APP_DIR, MODELS_DIR, EMBEDDINGS_DIR, DB_PATH
from .database_manager import delete_random_model, delete_model_by_dwx_path, insert_or_update_model

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

class NewModel(BaseModel):
    name: str
    path: str
    jpg_path: str = None

class UpdateDirectories(BaseModel):
    directories: List[str]

# --- LIFECYCLE EVENTS ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, image_paths

    print("🚀 Server startup initiated...")

    # Handle offline deletions
    deletions_log_path = CPP_APP_DIR / "deletions.log"

    if deletions_log_path.exists():
        print("🔍 Found deletions.log. Processing offline deletions...")
        with open(deletions_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                op, value = parts
                if op == "path":
                    print(f"  - Deleting by path: {value}")
                    delete_model_by_dwx_path(DB_PATH, value)
                elif op == "random":
                    print(f"  - Deleting random model")
                    delete_random_model(DB_PATH)

        os.remove(deletions_log_path)
        print("✅ Finished processing deletions.log. Rebuilding index...")
        rebuild_index_from_db(DB_PATH)


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
            local_files_only=True
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
index, image_paths = None, None  


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

@app.post("/rebuild-index")
def rebuild_index_endpoint():
    """
    Endpoint to completely rebuild the FAISS index from the database.
    This is useful after deleting models.
    """
    print("📥 Received request to rebuild the entire FAISS index.")
    start_time = time.time()
    try:
        # Call the new rebuild function
        message = rebuild_index_from_db(DB_PATH)

        # Critical: Reload the index into the running server's memory
        global index, image_paths
        index, image_paths = load_index_and_paths()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Index rebuild completed in {elapsed:.2f} seconds.")
        
        return JSONResponse(content={"message": f"{message} (Took {elapsed:.2f}s)"})
        
    except Exception as e:
        print(f"❌ Error during index rebuild operation: {str(e)}")
        traceback.print_exc()
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
            prompt.text.lower().strip(),
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
def find_similar_images_by_image(file: UploadFile = File(...), top_k: int = 10):
    """
    Endpoint wyszukiwania graficznego (Image-to-Image).
    Uruchamiany synchronicznie w Thread Poolu, dzięki czemu ciężka inferencja ONNX
    nie blokuje innych żądań płynących do serwera.
    """
    global index, image_paths
    print(f"Received image search request (top_k={top_k})")
    start_time = time.time()
    
    if index is None:
        raise HTTPException(status_code=400, detail="Index not loaded.")

    # Upewniamy się, że katalog tymczasowy istnieje na dysku
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        file_bytes = file.file.read()
        
        with open(temp_path, "wb") as buffer:
            buffer.write(file_bytes)

        pixel_values = preprocess_image(str(temp_path))
        image_embedding = get_image_embedding(pixel_values)
        
        results = search_similar_images(image_embedding, index, image_paths, top_k)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Image search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"❌ Image search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Pancerne czyszczenie dysku SSD z plików tymczasowych
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as unlink_err:
                print(f"⚠️ Nie udalo sie usunac pliku tymczasowego: {unlink_err}")
    
@app.get("/")
def root():
    return {"status": "server is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --- MODEL MANAGEMENT ENDPOINTS ---
@app.post("/add-model")
def add_model_endpoint(model: NewModel):
    """
    Adds a new model to the database and triggers an embedding update.
    """
    print(f"📥 Received request to add new model: {model.name}")
    try:
        model_data = {"name": model.name, "dwx_path": model.path, "jpg_path": model.jpg_path}
        insert_or_update_model(DB_PATH, model_data)
        
        # After adding, we need to update the embeddings and the index
        update_embeddings_from_db(DB_PATH)
        global index, image_paths
        index, image_paths = load_index_and_paths()

        return JSONResponse(content={"message": f"Model '{model.name}' added successfully and index updated."})
    except Exception as e:
        print(f"❌ Error during model addition: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/model")
def delete_model_by_path_endpoint(dwx_path: str):
    """
    Deletes a model by its unique dwx_path from the database.
    """
    print(f"📥 Received request to delete model with path: {dwx_path}")
    try:
        success = delete_model_by_dwx_path(DB_PATH, dwx_path)
        if success:
            return JSONResponse(content={"message": f"Model with path '{dwx_path}' deleted successfully."})
        else:
            raise HTTPException(status_code=404, detail=f"Model with path '{dwx_path}' not found.")
    except Exception as e:
        print(f"❌ Error during model deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/random")
def delete_random_model_endpoint():
    """
    Deletes a random model from the database.
    """
    print("📥 Received request to delete a random model.")
    try:
        deleted_id = delete_random_model(DB_PATH)
        if deleted_id is not None:
            return JSONResponse(content={"message": f"Randomly deleted model with ID: {deleted_id}"})
        else:
            raise HTTPException(status_code=404, detail="No models found in the database to delete.")
    except Exception as e:
        print(f"❌ Error during random model deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN RUNNER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)