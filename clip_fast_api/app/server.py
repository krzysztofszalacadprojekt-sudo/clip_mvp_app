import os
import time
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from dotenv import load_dotenv
from .image_preprocessor import preprocess_image
# 🚀 POPRAWKA: Zaimportowano _get_active_session, aby móc pobrać instancję DirectML
from .clip_model import get_text_embedding as get_clip_text_embedding, get_image_embedding, _get_active_session
from app import config
from .embedding_store import (
    create_and_save_embeddings,
    load_index_and_paths,
    search_similar_images,
    update_embeddings_from_db,
    reset_stored_embeddings,
    search_similar_metadata_only, # 🚀 Dodaj to
    search_similar_hybrid,        # 🚀 Dodaj to
    search_similar_images_from_db,
    embed_images_batch,
    embed_texts_batch             # Do bezpośredniego kodowania zapytania
)
from .config import DB_PATH
from threading import Thread
import traceback
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Set HF Token if needed for the tokenizer loading logic
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    print("⚠️  HF_TOKEN environment variable not set. This may be required for some models.")

# --- SCHEMAS ---

class TextPrompt(BaseModel):
    text: str
    top_k: int = 5

class UpdateDirectories(BaseModel):
    directories: List[str]

# Dopisz pod klasą TextPrompt:
class HybridTextPrompt(BaseModel):
    text: str
    top_k: int = 5
    alpha: float = 0.8  # 🚀 Waga dla obrazu (np. 0.4 oznacza 40% obraz, 60% tekst metadata)

class RebuildEmbeddingsRequest(BaseModel):
    images: bool = True
    text: bool = True

# --- LIFECYCLE EVENTS ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, image_paths

    print("🚀 Server startup initiated...")

    Path("temp").mkdir(exist_ok=True)

    try:
        # ---------------------------
        # STEP 1: TOKENIZER
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
        # STEP 2: FAISS
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
        # 🚀 STEP 3: ONNX DIRECTML TEXT SESSION INITIALIZATION
        # ---------------------------
        print("STEP 3: Loading ONNX Text Session via DirectML...")
        start_session = time.time()
        
        print("STEP 2.5: Loading DirectML Text Session...")
        app.state.text_session = _get_active_session("text")
        print("✅ DirectML Text Session hooked successfully.")
        
        print(f"✅ ONNX Text Session ready in {time.time() - start_session:.2f}s")

        # ---------------------------
        # STEP 4: BACKGROUND AUTOMATIC SYNC
        # --------------------------
        def auto_sync_worker():
            global index, image_paths
            try:
                print("🔍 [Background Sync] Checking SQLite for new models missing embeddings...")
                sync_start = time.time()
                
                # Przetwarzanie bazy danych SigLIP-em w osobnym wątku - sesja pobierana z app.state
                message = update_embeddings_from_db(DB_PATH, app.state.tokenizer, app.state.text_session)
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
        app.state.text_session = None

    yield

    print("🛑 Server shutdown.")

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)
index, image_paths = None, None  

@app.post("/sync")
async def sync_faiss_index(request: Request):
    try:
        # Pobieramy referencje do silników załadowanych w lifespan przy starcie
        active_tokenizer = app.state.tokenizer 
        active_text_session = app.state.text_session
        
        # Przekazujemy komplet narzędzi do skryptu bazy danych
        message = update_embeddings_from_db(str(config.DB_PATH), active_tokenizer, active_text_session)
        
        return {"status": "success", "detail": message}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/rebuild-embeddings")
async def rebuild_embeddings(request_body: RebuildEmbeddingsRequest, request: Request):
    """
    Clears selected stored vectors and regenerates them from the current ONNX output selection.
    Model metadata remains untouched.
    """
    global index, image_paths

    try:
        active_tokenizer = request.app.state.tokenizer
        active_text_session = request.app.state.text_session

        if active_tokenizer is None or active_text_session is None:
            raise HTTPException(status_code=500, detail="AI Engine components are uninitialized.")

        reset_message = reset_stored_embeddings(
            str(config.DB_PATH),
            reset_images=request_body.images,
            reset_text=request_body.text,
        )
        rebuild_message = update_embeddings_from_db(str(config.DB_PATH), active_tokenizer, active_text_session)

        index, image_paths = load_index_and_paths()

        return {
            "status": "success",
            "reset": reset_message,
            "rebuild": rebuild_message,
        }
    except Exception as e:
        print(f"❌ Rebuild embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/find-similar-images-by-text-metadata")
def find_similar_images_by_text_metadata(prompt: TextPrompt, request: Request):
    """Trasa wyszukująca meble WYŁĄCZNIE po wektorach metadanych tekstowych."""
    print(f"🔎 Received metadata-only text search request: {prompt.text} (top_k={prompt.top_k})")
    start_time = time.time()
    
    try:
        active_tokenizer = request.app.state.tokenizer
        active_text_session = request.app.state.text_session
        
        if active_tokenizer is None or active_text_session is None:
            raise HTTPException(status_code=500, detail="AI Models/Tokenizers are not initialized.")

        # Generujemy wektor dla wpisanej frazy użytkownika przy użyciu naszej bezpiecznej metody
        query_embedding = embed_texts_batch([prompt.text], active_tokenizer, active_text_session)
        
        # Wykonujemy wyszukiwanie bezpośrednio w bazie danych SQLite
        results = search_similar_metadata_only(query_embedding, str(config.DB_PATH), prompt.top_k)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Metadata text search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"❌ Metadata search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find-similar-images-by-hybrid-search")
def find_similar_images_by_hybrid_search(prompt: HybridTextPrompt, request: Request):
    """Trasa łącząca siłę oka AI (FAISS) oraz rozumu AI (metadane SQLite)."""
    print(f"🎛️ Received HYBRID search request: {prompt.text} (alpha={prompt.alpha}, top_k={prompt.top_k})")
    start_time = time.time()

    try:
        active_tokenizer = request.app.state.tokenizer
        active_text_session = request.app.state.text_session
        
        if active_tokenizer is None or active_text_session is None:
            raise HTTPException(status_code=500, detail="AI Engine components are uninitialized.")

        # Generujemy wektor zapytania
        query_embedding = embed_texts_batch([prompt.text], active_tokenizer, active_text_session)
        
        # Wywołujemy silnik fuzji wagowej z embedding_store
        results = search_similar_hybrid(
            query_embedding=query_embedding,
            db_path=str(config.DB_PATH),
            alpha=prompt.alpha,
            top_k=prompt.top_k
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ Hybrid score-fusion search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        message = update_embeddings_from_db(DB_PATH, app.state.tokenizer, app.state.text_session)
        
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
async def find_similar_images_by_image(file: UploadFile = File(...), top_k: int = 5):
    print(f"Received image search request (top_k={top_k})")
    start_time = time.time()

    try:
        temp_path = None
        try:
            temp_path = Path("temp") / file.filename
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())

            image_embedding, valid_paths = embed_images_batch([str(temp_path)])
            if image_embedding.shape[0] == 0 or not valid_paths:
                raise HTTPException(status_code=400, detail="Could not generate an embedding for the uploaded image.")
            
            results = search_similar_images_from_db(image_embedding, str(config.DB_PATH), top_k)
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
