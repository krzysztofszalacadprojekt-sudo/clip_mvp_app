from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List

from . import config
from .text_preprocessor import preprocess_text
from .image_preprocessor import preprocess_image
from .clip_model import get_text_embedding, get_image_embedding
from .embedding_store import (
    create_and_save_embeddings,
    load_index_and_paths,
    search_similar_images,
    update_embeddings,
)

app = FastAPI()

index, image_paths = None, None  # Initialize to None, load on first request or via endpoint

class TextPrompt(BaseModel):
    text: str
    top_k: int = 5

class UpdateDirectories(BaseModel):
    directories: List[str]

@app.post("/create-embeddings")
def create_embeddings_endpoint():
    """
    Endpoint to create and save image embeddings.
    """
    try:
        create_and_save_embeddings()
        # Reload the index and paths after creation
        global index, image_paths
        index, image_paths = load_index_and_paths()
        return JSONResponse(
            content={"message": "Embeddings created and saved successfully."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-embeddings")
def update_embeddings_endpoint(update_dirs: UpdateDirectories):
    """
    Endpoint to update embeddings from a list of directories.
    """
    try:
        message = update_embeddings(update_dirs.directories)
        # Reload the index and paths after update
        global index, image_paths
        index, image_paths = load_index_and_paths()
        return JSONResponse(
            content={"message": message}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-images-by-text")
def find_similar_images_by_text(prompt: TextPrompt):
    """
    Endpoint to find similar images based on a text prompt.
    """
    global index, image_paths
    if index is None or image_paths is None:
        index, image_paths = load_index_and_paths()
        if index is None or image_paths is None:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not created. Please run /create-embeddings or /update-embeddings first.",
            )
    
    try:
        inputs = preprocess_text(prompt.text)
        text_embedding = get_text_embedding(inputs['input_ids'])
        
        results = search_similar_images(
            text_embedding, index, image_paths, prompt.top_k
        )
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find-similar-images-by-image")
async def find_similar_images_by_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Endpoint to find similar images based on an image prompt.
    """
    if index is None or image_paths is None:
        raise HTTPException(
            status_code=400,
            detail="Embeddings not created. Please run /create-embeddings or /update-embeddings first.",
        )

    try:
        # Save the uploaded file temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        pixel_values = preprocess_image(str(file_path))
        image_embedding = get_image_embedding(pixel_values)
        
        results = search_similar_images(
            image_embedding, index, image_paths, top_k
        )

        # Clean up the temporary file
        file_path.unlink()

        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """
    On startup, create a temp directory if it doesn't exist.
    Note: Embedding updates are now handled via explicit API calls.
    """
    Path("temp").mkdir(exist_ok=True)
    print("FastAPI server started. Use /create-embeddings or /update-embeddings to build the index.")
