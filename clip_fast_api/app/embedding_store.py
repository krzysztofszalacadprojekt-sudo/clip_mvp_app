import faiss
import numpy as np
import json
from pathlib import Path
from .config import FAISS_INDEX_PATH, IMAGE_PATHS_LIST, IMAGES_DIR
from PIL import Image
from typing import List
import concurrent.futures
from . import image_preprocessor
from .clip_model import get_image_embeddings_batch
import os
import traceback
import psutil
import platform
import subprocess

# --- Dynamic Batch Size Calculation ---

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32
RAM_TARGET_FRACTION = 0.4  # Be conservative with RAM
MEMORY_PER_IMAGE_BYTES = 50 * 1024 * 1024  # Estimate ~50MB per image in flight (tensors + arrays)
BASE_MODEL_VRAM_BYTES = 1.2 * 1024 * 1024 * 1024  # ~1.2 GB for CLIP ViT-B/32
ONNX_VRAM_PER_IMAGE_BYTES = 15 * 1024 * 1024  # ~15MB intermediate activation memory per image

def _get_available_vram_bytes():
    """Attempts to detect GPU VRAM natively on Windows. Returns 4GB fallback if it fails."""
    default_vram = 4 * 1024**3  # 4 GB fallback
    if platform.system() == "Windows":
        try:
            # Use powershell to query Windows for installed GPU memory
            cmd = 'powershell -command "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty AdapterRAM"'
            output = subprocess.check_output(cmd, shell=True, text=True).strip()
            vrams = [int(v.strip()) for v in output.split('\n') if v.strip().isdigit()]
            if vrams:
                return max(vrams)
        except Exception as e:
            print(f"VRAM detection failed ({e}). Using default 4GB.")
    return default_vram

def _get_dynamic_batch_size():
    try:
        # 1. Check RAM
        available_ram = psutil.virtual_memory().available
        available_ram_gb = available_ram / (1024 ** 3)
        
        # CRITICAL HARD FLOOR: If total available RAM is critically low, 
        # force batch_size=1 instantly. Do not let math dictate a higher number.
        if available_ram_gb < 2.5:
            print(f"⚠️ Critically low RAM detected ({available_ram_gb:.2f} GB). Forcing safe batch size of 1.")
            return 1

        usable_ram = available_ram * RAM_TARGET_FRACTION
        ram_batch_size = int(usable_ram // MEMORY_PER_IMAGE_BYTES)

        # 2. Calculate VRAM constraints
        # Ensure your helper function ONLY returns DEDICATED VRAM, not shared system VRAM!
        vram_bytes = _get_available_vram_bytes() 
        usable_vram = vram_bytes * 0.7 
        available_vram_for_batch = max(0, usable_vram - BASE_MODEL_VRAM_BYTES)
        vram_batch_size = int(available_vram_for_batch // ONNX_VRAM_PER_IMAGE_BYTES)

        # Take the most restrictive limit
        calculated_batch_size = min(ram_batch_size, vram_batch_size)

        # Apply strict clamping bounds
        dynamic_batch_size = max(MIN_BATCH_SIZE, min(calculated_batch_size, MAX_BATCH_SIZE))
        
        print(f"Available RAM: {available_ram_gb:.2f} GB | Detected VRAM: {vram_bytes / (1024 ** 3):.2f} GB")
        print(f"RAM constraint: {ram_batch_size} max | VRAM constraint: {vram_batch_size} max")
        print(f"🚀 Dynamic batch size safely set to: {dynamic_batch_size}")
        return dynamic_batch_size

    except Exception as e:
        print(f"Resource detection failed ({e}). Falling back to default batch size of 1.")
        return 1


def embed_images_batch(image_paths: List[str]):
    """
    Compute L2-normalized CLIP image embeddings for a list of image paths using batching.
    Returns a (N, D) numpy array and a list of valid paths.
    """
    feats = []
    valid_paths = []

    batch_size = _get_dynamic_batch_size()
    total_images = len(image_paths)
    processed_images = 0
    print(f"Starting to process {total_images} images in batches of up to {batch_size}...")

    def load_image(p):
        try:
            # 'with' ensures file handles are closed eagerly. 
            # 'convert' forces the pixel data to be fully loaded into memory.
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
            return p, img_rgb, str(Path(p).expanduser().resolve())
        except Exception as e:
            print(f"Skipping {p}: {e}")
            return p, None, None

    image_preprocessor._init_processor()

    # Process chunks one by one so we don't load thousands of images into RAM at once
    for i in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[i : i + batch_size]
        batch_imgs = []
        batch_valid_paths = []

        # Load only the current batch of images using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for p, img, resolved_path in executor.map(load_image, chunk_paths):
                if img is not None:
                    batch_imgs.append(img)
                    batch_valid_paths.append(resolved_path)

        if batch_imgs:
            inputs = image_preprocessor.processor(images=batch_imgs, return_tensors="np")
            image_features = get_image_embeddings_batch(inputs["pixel_values"])
            faiss.normalize_L2(image_features)
            
            feats.append(image_features)
            valid_paths.extend(batch_valid_paths)
            
            # Explicitly free memory before loading the next chunk
            del batch_imgs
            del inputs
            
        processed_images += len(chunk_paths)
        progress_percent = (processed_images / total_images) * 100
        print(f"Progress: {processed_images}/{total_images} images processed ({progress_percent:.1f}%)")

    if not feats:
        return np.zeros((0, 0), dtype=np.float32), []

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    return feats, valid_paths

def create_and_save_embeddings(image_paths: List[str] = None):
    """
    Creates embeddings for all images in the IMAGES_DIR using batch processing,
    builds a FAISS index, and saves it to disk.
    If image_paths is provided, it uses that list instead of searching the directory.
    """
    if image_paths is None:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = [p for p in Path(IMAGES_DIR).rglob("*") if p.suffix.lower() in valid_exts]

    # Normalize all paths (absolute + resolved) so comparisons are consistent
    image_paths = [Path(p).expanduser().resolve() for p in image_paths]
    
    if not image_paths:
        print("No images found to embed.")
        return

    embeddings, valid_paths = embed_images_batch(image_paths)

    if embeddings.shape[0] == 0:
        print("No embeddings were generated.")
        return

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(IMAGE_PATHS_LIST, "w", encoding="utf-8") as f:
        json.dump(valid_paths, f)
        
def load_index_and_paths():
    """
    Loads the FAISS index and the list of image paths.

    If duplicates are detected in the stored paths, it rebuilds the index from the
    unique set of paths to avoid returning repeated results.
    """
    if not FAISS_INDEX_PATH.exists() or not IMAGE_PATHS_LIST.exists():
        return None, None
        
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(IMAGE_PATHS_LIST, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    # Normalize stored paths so comparisons remain consistent across runs
    normalized_paths = [str(Path(p).expanduser().resolve()) for p in image_paths]

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in normalized_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if len(unique_paths) != len(normalized_paths):
        # Rebuild index from unique list to remove duplicates
        create_and_save_embeddings(unique_paths)
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        normalized_paths = unique_paths

    return index, normalized_paths

def search_similar_images(query_embedding, index, image_paths, top_k=5):
    """
    Searches the FAISS index for the most similar images.
    """
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        if i < len(indices[0]):
            results.append(
                {
                    "path": image_paths[indices[0][i]],
                    "distance": float(distances[0][i])
                }
            )
    return results

def update_embeddings(directories: List[str]):

    print("=== UPDATE EMBEDDINGS ===")
    print("Directories:", directories)

    index, image_paths = load_index_and_paths()

    existing_paths = (
        set(str(Path(p).expanduser().resolve())
            for p in image_paths)
        if image_paths
        else set()
    )

    normalized_dirs = []

    for directory in directories:
        try:
            d = Path(directory).expanduser().resolve()

            if not d.exists():
                print(f"⚠️ Directory not found: {directory}")
                continue

            normalized_dirs.append(d)

        except Exception as e:
            print(f"⚠️ Failed resolving path {directory}: {e}")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # =====================================================
    # CREATE NEW INDEX
    # =====================================================
    if index is None:

        all_image_paths = []

        print("Scanning images recursively...")

        for d in normalized_dirs:
            print(f"Scanning: {d}")

            for p in d.rglob("*"):

                if (
                    p.is_file()
                    and p.suffix.lower() in valid_exts
                ):
                    all_image_paths.append(str(p.resolve()))

        print(f"Found {len(all_image_paths)} images")

        if not all_image_paths:
            return "No images found to create index."

        create_and_save_embeddings(all_image_paths)

        return (
            f"Created new index with "
            f"{len(all_image_paths)} images."
        )

    # =====================================================
    # UPDATE EXISTING INDEX
    # =====================================================
    new_image_paths = []

    print("Looking for new images...")

    for d in normalized_dirs:

        for p in d.rglob("*"):

            if (
                p.is_file()
                and p.suffix.lower() in valid_exts
            ):

                try:
                    resolved = str(
                        p.expanduser().resolve()
                    )

                    if resolved not in existing_paths:
                        new_image_paths.append(resolved)
                        existing_paths.add(resolved)

                except Exception as e:
                    print(
                        f"⚠️ Failed resolving "
                        f"{p}: {e}"
                    )

    print(f"New images found: {len(new_image_paths)}")

    if not new_image_paths:
        return "No new images found."

    print("Generating embeddings...")

    try:
        new_embeddings, valid_new_paths = embed_images_batch(
            new_image_paths
        )

    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        traceback.print_exc()

        return "Embedding generation failed."

    if new_embeddings.shape[0] == 0:
        return "Could not generate embeddings."

    print("Adding embeddings to FAISS...")

    index.add(new_embeddings)

    print("Saving FAISS index...")

    # =====================================
    # ATOMIC SAVE (IMPORTANT)
    # =====================================
    tmp_index = str(FAISS_INDEX_PATH) + ".tmp"

    faiss.write_index(index, tmp_index)

    os.replace(
        tmp_index,
        str(FAISS_INDEX_PATH)
    )

    updated_image_paths = (
        image_paths + valid_new_paths
    )

    tmp_json = str(IMAGE_PATHS_LIST) + ".tmp"

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(
            updated_image_paths,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        tmp_json,
        IMAGE_PATHS_LIST
    )

    print("✅ Update finished.")

    return (
        f"Embeddings updated with "
        f"{len(valid_new_paths)} new images."
    )