import sys
import os
import time
from pathlib import Path
from typing import List

import uvicorn
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPTokenizerFast

from dotenv import load_dotenv

from .image_preprocessor import preprocess_image
from .clip_model import get_text_embedding as get_clip_text_embedding, get_image_embedding
from .config import BASE_DIR, TOKENIZER_DIR
from .embedding_store import (
    create_and_save_embeddings,
    load_index_and_paths,
    search_similar_images,
    update_embeddings,
)

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

        if not TOKENIZER_DIR.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {TOKENIZER_DIR}")

        app.state.tokenizer = CLIPTokenizerFast.from_pretrained(
            str(TOKENIZER_DIR),
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
    
    inputs = tokenizer(
        prompt.text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=77,
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
def update_embeddings_endpoint(update_dirs: UpdateDirectories):
    print(update_dirs.directories)
    start_time = time.time()
    try:
        message = update_embeddings(update_dirs.directories)
        global index, image_paths
        index, image_paths = load_index_and_paths()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Update embeddings completed in {elapsed:.2f} seconds.")
        return JSONResponse(content={"message": f"{message} (Took {elapsed:.2f}s)"})
    except Exception as e:
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
        tokenizer = request.app.state.tokenizer
        inputs = tokenizer(
            prompt.text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        
        text_embedding = get_clip_text_embedding(inputs)
        
        results = search_similar_images(text_embedding, index, image_paths, prompt.top_k)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Text search completed in {elapsed:.3f} seconds.")
        return JSONResponse(content=results)
    except Exception as e:
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