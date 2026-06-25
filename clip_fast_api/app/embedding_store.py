import faiss
import numpy as np
import json
from pathlib import Path
from .config import IMAGES_DIR, FAISS_INDEX_PATH, IMAGE_PATHS_LIST
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
import sqlite3
import time

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
        # 1. Odczyt systemowej pamięci RAM
        available_ram = psutil.virtual_memory().available
        available_ram_gb = available_ram / (1024 ** 3)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Krytyczna blokada dla bardzo słabych maszyn
        if available_ram_gb < 2.5:
            print(f"⚠️ Wykryto krytycznie niski poziom RAM ({available_ram_gb:.2f} GB). Wymuszenie bezpiecznego batch_size = 1.")
            return 1

        usable_ram = available_ram * RAM_TARGET_FRACTION
        ram_batch_size = int(usable_ram // MEMORY_PER_IMAGE_BYTES)

        # 2. Odczyt dedykowanej pamięci VRAM
        vram_bytes = _get_available_vram_bytes() 
        vram_gb = vram_bytes / (1024 ** 3)
        
        usable_vram = vram_bytes * 0.7 
        available_vram_for_batch = max(0, usable_vram - BASE_MODEL_VRAM_BYTES)
        vram_batch_size = int(available_vram_for_batch // ONNX_VRAM_PER_IMAGE_BYTES)

        # 3. DYNAMICZNE USTALANIE MAKSYMALNEGO SUFITU (Hardware Tiering)
        # Urealnione progi dopasowane do rzeczywistych specyfikacji GPU
        if vram_gb >= 11.0 and total_ram_gb >= 23.0:
            # Klasa Workstation (karty 12GB, 16GB, 24GB VRAM + 32GB+ RAM)
            adaptive_max_batch = 128
            tier_name = "High-End Workstation Rig"
            
        elif vram_gb >= 5.0 and total_ram_gb >= 11.0:
            # Klasa Średnia/Produkcyjna (karty 6GB, 8GB VRAM + 16GB RAM)
            # Teraz Twoja karta (nawet 6GB) bez problemu wpadnie tutaj!
            adaptive_max_batch = 64
            tier_name = "Standard Production Rig"
            
        else:
            # Klasa podstawowa / biurowa / fallback przy błędzie detekcji
            # Podnosimy sufit z 16 na 32, skoro sprawdziłeś, że 32 działa u Ciebie idealnie.
            adaptive_max_batch = 32
            tier_name = "Safe Guardrail Mode"

        # 4. Wybór najbardziej rygorystycznego ograniczenia z obliczonych matematycznie
        calculated_batch_size = min(ram_batch_size, vram_batch_size)

        # 5. Spięcie wyniku w bezpiecznych, wyliczonych dynamicznie widełkach
        dynamic_batch_size = max(MIN_BATCH_SIZE, min(calculated_batch_size, adaptive_max_batch))
        
        print(f"--- Wykryto profil sprzętowy: {tier_name} ---")
        print(f"Dostępny RAM: {available_ram_gb:.2f} GB | Wykryty VRAM: {vram_gb:.2f} GB")
        print(f"Matematyczny limit RAM: {ram_batch_size} | Limit VRAM: {vram_batch_size} | Dynamiczny Sufit Klasy: {adaptive_max_batch}")
        print(f"🚀 Bezpieczny rozmiar partii (Batch Size) ustawiony na: {dynamic_batch_size}")
        return dynamic_batch_size

    except Exception as e:
        print(f"Automatyczna detekcja zasobów nie powiodła się ({e}). Powrót do bezpiecznego bezpiecznika = 1.")
        return 1

def embed_images_batch(image_paths: List[str]) -> tuple[np.ndarray, List[str]]:
    """
    Przetwarza paczkę ścieżek obrazów na embeddingi SigLIP 2 przez ONNX Runtime + DirectML.
    Bezpiecznie zasila zunifikowane grafy makietami wejść tekstowych i celuje w węzeł wizualny.
    """
    
    if not image_paths:
        return np.empty((0, 768), dtype=np.float32), []

    # Pobieramy instancje sesji (zakładamy, że serwer używa zunifikowanej aktywnej sesji)
    # W server.py przypisaliśmy ją do app.state.text_session, upewnij się, że masz do niej dostęp
    # Może to być też wspólna zmienna globalna shared_session z clip_model
    from .clip_model import _get_active_session
    session = _get_active_session("visual") 

    expected_inputs = [node.name for node in session.get_inputs()]
    expected_outputs = [node.name for node in session.get_outputs()]

    if "image_embeds" in expected_outputs:
        target_output = "image_embeds"
    else:
        target_output = expected_outputs[0]
        for output_name in expected_outputs:
            name_lower = output_name.lower()
            if (
                ("embed" in name_lower or "feature" in name_lower)
                and ("image" in name_lower or "vision" in name_lower)
            ):
                target_output = output_name
                break

    batch_size = max(1, min(_get_dynamic_batch_size(), 32))
    feats = []
    valid_paths = []

    def run_small_batch(batch_paths: List[str]) -> tuple[np.ndarray, List[str]]:
        batch_imgs = []
        batch_valid_paths = []

        for p in batch_paths:
            try:
                if not Path(p).exists():
                    continue

                with Image.open(p) as img:
                    batch_imgs.append(img.convert("RGB"))
                batch_valid_paths.append(str(p))
            except Exception as img_err:
                print(f"Nie można wczytać obrazu {p}: {img_err}")

        if not batch_imgs:
            return np.empty((0, 768), dtype=np.float32), []

        if image_preprocessor.processor is None:
            image_preprocessor._init_processor()

        inputs = image_preprocessor.processor(images=batch_imgs, return_tensors="np")

        onnx_inputs = {
            "pixel_values": inputs["pixel_values"].astype(np.float32)
        }

        if "input_ids" in expected_inputs:
            onnx_inputs["input_ids"] = np.zeros((len(batch_valid_paths), 64), dtype=np.int64)
            
        if "attention_mask" in expected_inputs:
            onnx_inputs["attention_mask"] = np.ones((len(batch_valid_paths), 64), dtype=np.int64)

        raw_outputs = session.run([target_output], onnx_inputs)
        image_features = raw_outputs[0]

        if image_features.ndim == 2 and image_features.shape[0] != len(batch_valid_paths) and image_features.shape[1] == len(batch_valid_paths):
            image_features = image_features.T
        if image_features.ndim != 2 or image_features.shape[0] != len(batch_valid_paths):
            raise ValueError(
                f"Selected image output '{target_output}' has invalid shape {image_features.shape}; "
                f"expected ({len(batch_valid_paths)}, embedding_dim)."
            )

        norms = np.linalg.norm(image_features, axis=-1, keepdims=True)
        normalized_features = image_features / np.where(norms == 0, 1e-12, norms)

        return normalized_features.astype(np.float32), batch_valid_paths

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        try:
            batch_embeddings, batch_valid_paths = run_small_batch(batch_paths)
        except Exception as batch_err:
            if len(batch_paths) == 1:
                print(f"Pominięto obraz po błędzie ONNX {batch_paths[0]}: {repr(batch_err)}")
                continue

            print(f"Batch obrazów {i + 1}-{i + len(batch_paths)} nie zmieścił się w pamięci. Schodzę do pojedynczych obrazów.")
            for single_path in batch_paths:
                try:
                    batch_embeddings, batch_valid_paths = run_small_batch([single_path])
                except Exception as single_err:
                    print(f"Pominięto obraz po błędzie ONNX {single_path}: {repr(single_err)}")
                    continue

                if batch_embeddings.shape[0] > 0:
                    feats.append(batch_embeddings)
                    valid_paths.extend(batch_valid_paths)
            continue

        if batch_embeddings.shape[0] > 0:
            feats.append(batch_embeddings)
            valid_paths.extend(batch_valid_paths)

    if not feats:
        return np.empty((0, 768), dtype=np.float32), []

    return np.concatenate(feats, axis=0).astype(np.float32), valid_paths

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
    Ładuje indeks FAISS oraz listę ścieżek z dysku do pamięci RAM.
    Wersja produkcyjna: bezpieczna dla ścieżek względnych (bez .resolve())
    """
    from .config import FAISS_INDEX_PATH, IMAGE_PATHS_LIST

    if not FAISS_INDEX_PATH.exists() or not IMAGE_PATHS_LIST.exists():
        return None, None
        
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(IMAGE_PATHS_LIST, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    normalized_paths = [p.replace("/", "\\").strip() for p in image_paths]

    return index, normalized_paths

def search_similar_images(query_embedding, index, image_paths, top_k=5):
    """
    Bezpieczna funkcja przeszukiwania indeksu FAISS.
    Odporna na błędy synchronizacji (IndexError) oraz puste wyniki (-1).
    """
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    # Sprawdzenie czy FAISS w ogóle cokolwiek zwrócił
    if len(indices) == 0 or len(indices[0]) == 0:
        return results
        
    for i in range(top_k):
        if i < len(indices[0]):
            idx = indices[0][i]
            
            # FAISS zwraca -1, jeśli indeks jest pusty lub szukamy więcej wyników niż jest w bazie
            if idx == -1:
                continue
                
            # Defensywne sprawdzenie granic tablicy (Zabezpieczenie przed list index out of range)
            if 0 <= idx < len(image_paths):
                path_to_add = IMAGES_DIR / image_paths[idx]
                if path_to_add.exists():
                    results.append(
                        {
                            "path": str(path_to_add),
                            "distance": float(distances[0][i])
                        }
                    )
            else:
                print(f"⚠️ [Defensive Guard] FAISS zwrócił indeks {idx}, ale image_paths ma tylko {len(image_paths)} elementów! Pomijam.")
                
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

def _resolve_db_image_path(db_path_value: str) -> Path:
    """Return an absolute image path for a path stored in SQLite."""
    path_obj = Path(db_path_value)
    if path_obj.is_absolute():
        return path_obj
    return IMAGES_DIR / path_obj


def _safe_json_vector(raw_value: str) -> np.ndarray | None:
    if not raw_value:
        return None
    try:
        return np.array(json.loads(raw_value), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Skipping corrupt vector from SQLite: {exc}")
        return None


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_v = values.min()
    max_v = values.max()
    span = max_v - min_v
    if span <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_v) / span


def _normalized_flat_vector(vector: np.ndarray) -> np.ndarray:
    flat = vector.astype("float32").flatten()
    norm = np.linalg.norm(flat)
    if norm > 1e-12:
        return flat / norm
    return flat


def reset_stored_embeddings(db_path: str, reset_images: bool = True, reset_text: bool = True) -> str:
    """
    Clears only stored embedding vectors and flags. Metadata/model rows stay intact.
    The next update_embeddings_from_db run will regenerate cleared vectors.
    """
    if not reset_images and not reset_text:
        return "No embedding columns selected for reset."

    assignments = []
    if reset_images:
        assignments.extend(["image_vector = NULL", "image_embedding_exists = 0"])
    if reset_text:
        assignments.extend(["text_vector = NULL", "text_embedding_exists = 0"])

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE models SET {', '.join(assignments)}")
        affected = cursor.rowcount
        conn.commit()
        return f"Reset embedding columns for {affected} model rows."
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def search_similar_metadata_only(query_embedding: np.ndarray, db_path: str, top_k: int = 5) -> list:
    """
    Wyszukuje meble wyłącznie na podstawie podobieństwa semantycznego do metadanych tekstowych z SQLite.
    Ignoruje indeks FAISS (obrazki).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Pobieramy z bazy wszystkie modele, które posiadają wygenerowany wektor tekstu
    cursor.execute("""
        SELECT id, name, manufacturer, jpg_path, text_vector
        FROM models
        WHERE text_vector IS NOT NULL AND jpg_path IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    results = []
    q_vec = _normalized_flat_vector(query_embedding)
    
    for row in rows:
        # Dekodujemy zapisany ciąg JSON z bazy z powrotem do tablicy NumPy
        t_vec = _safe_json_vector(row["text_vector"])
        if t_vec is None:
            continue
        t_vec = _normalized_flat_vector(t_vec)
        
        # Obliczamy podobieństwo cosinusowe (iloczyn skalarny dla wektorów znormalizowanych L2)
        similarity = float(np.dot(q_vec, t_vec))
        
        path_to_add = _resolve_db_image_path(row["jpg_path"])
        results.append({
            "id": row["id"],
            "name": row["name"],
            "manufacturer": row["manufacturer"],
            "path": str(path_to_add),
            "distance": similarity  # W naszej architekturze 'distance' to miara podobieństwa (im więcej tym lepiej)
        })
        
    # Sortujemy od najwyższego podobieństwa tekstowego
    results.sort(key=lambda x: x["distance"], reverse=True)
    return results[:top_k]

def search_similar_images_from_db(query_embedding: np.ndarray, db_path: str, top_k: int = 5) -> list:
    """
    Search image embeddings stored in SQLite. This is the source of truth for
    add/delete workflows; FAISS can still exist as a rebuildable cache.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, manufacturer, jpg_path, image_vector
        FROM models
        WHERE image_vector IS NOT NULL AND jpg_path IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    q_vec = _normalized_flat_vector(query_embedding)
    results = []

    for row in rows:
        image_vec = _safe_json_vector(row["image_vector"])
        if image_vec is None:
            continue
        image_vec = _normalized_flat_vector(image_vec)

        results.append({
            "id": row["id"],
            "name": row["name"],
            "manufacturer": row["manufacturer"],
            "path": str(_resolve_db_image_path(row["jpg_path"])),
            "distance": float(np.dot(q_vec, image_vec)),
        })

    results.sort(key=lambda x: x["distance"], reverse=True)
    return results[:top_k]


def search_similar_hybrid(query_embedding: np.ndarray, db_path: str, alpha: float = 0.35, top_k: int = 5) -> list:
    """
    Wyszukiwanie hybrydowe z dynamicznym skalowaniem Min-Max dla obu modalności.
    Sprowadza wyniki tekstu i obrazu do przedziału [0, 1] przed nałożeniem wag Alpha/Beta.
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    beta = 1.0 - alpha
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, manufacturer, jpg_path, image_vector, text_vector
        FROM models
        WHERE jpg_path IS NOT NULL
          AND (image_vector IS NOT NULL OR text_vector IS NOT NULL)
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    raw_text_scores = []
    raw_image_scores = []
    metadata_list = []
    
    q_vec = _normalized_flat_vector(query_embedding)
    
    for row in rows:
        t_vec = _safe_json_vector(row["text_vector"])
        i_vec = _safe_json_vector(row["image_vector"])

        score_text = float(np.dot(q_vec, _normalized_flat_vector(t_vec))) if t_vec is not None else 0.0
        score_image = float(np.dot(q_vec, _normalized_flat_vector(i_vec))) if i_vec is not None else 0.0
        
        raw_text_scores.append(score_text)
        raw_image_scores.append(score_image)
        metadata_list.append({
            "id": row["id"],
            "name": row["name"],
            "manufacturer": row["manufacturer"],
            "path": row["jpg_path"]
        })

    np_text = np.array(raw_text_scores, dtype=np.float32)
    np_image = np.array(raw_image_scores, dtype=np.float32)

    norm_text_scores = _minmax(np_text)
    norm_image_scores = _minmax(np_image)
    final_hybrid_scores = (alpha * norm_image_scores) + (beta * norm_text_scores)

    results = []
    for idx, meta in enumerate(metadata_list):
        path_to_add = _resolve_db_image_path(meta["path"])
        results.append({
            "id": meta["id"],
            "name": meta["name"],
            "manufacturer": meta["manufacturer"],
            "path": str(path_to_add),
            "distance": float(final_hybrid_scores[idx]), # Wykorzystywane przez C++ jako miara dopasowania
            "debug_image_score_raw": float(np_image[idx]),
            "debug_text_score_raw": float(np_text[idx])
        })
        
    results.sort(key=lambda x: x["distance"], reverse=True)
    return results[:top_k]

def embed_texts_batch(texts: List[str], tokenizer, text_session) -> np.ndarray:
    """
    Koduje paczkę tekstów na embeddingi SigLIP 2.
    Powrót do klasycznego, przewidywalnego wyciągania węzłów wyjściowych z sesji ONNX.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    MICRO_BATCH_SIZE = 32
    all_features = []

    # 1. Pobieramy surowe nazwy wejść i wyjść dokładnie tak, jak widzi je sesja ONNX
    expected_inputs = [node.name for node in text_session.get_inputs()]
    expected_outputs = [node.name for node in text_session.get_outputs()]

    # 🚀 JAWNA DIAGNOSTYKA: Zobaczysz w konsoli dokładne nazwy wyjść Twojego modelu
    print(f"📊 [ONNX DIAGNOSTIC] Wykryte węzły wyjściowe w modelu: {expected_outputs}")

    # 2. Powrót do starej logiki: Wybieramy właściwy węzeł bez przekombinowanych automatów
    if "text_embeds" in expected_outputs:
        target_output = "text_embeds"
    elif len(expected_outputs) > 1:
        target_output = next(
            (
                output_name for output_name in expected_outputs
                if "text" in output_name.lower() and ("embed" in output_name.lower() or "feature" in output_name.lower())
            ),
            expected_outputs[1],
        )
    else:
        target_output = expected_outputs[0]

    print(f"🎯 Celowanie zablokowane na węźle: '{target_output}'")

    try:
        from . import image_preprocessor
        from PIL import Image
        
        if image_preprocessor.processor is None:
            image_preprocessor._init_processor()
            
        dummy_img = Image.new("RGB", (10, 10))
        dummy_features = image_preprocessor.processor(images=[dummy_img], return_tensors="np")
        actual_vision_shape = dummy_features["pixel_values"].shape

        # 3. Pętla przetwarzania (Micro-batching chroniący przed brakiem pamięci VRAM)
        for b in range(0, len(texts), MICRO_BATCH_SIZE):
            chunk_texts = texts[b : b + MICRO_BATCH_SIZE]
            
            inputs = tokenizer(
                chunk_texts, 
                padding=True, 
                truncation=True, 
                max_length=64, 
                return_tensors="np"
            )

            # Mapowanie wejść
            onnx_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64)
            }

            if "attention_mask" in expected_inputs:
                if "attention_mask" in inputs:
                    onnx_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
                else:
                    onnx_inputs["attention_mask"] = np.ones_like(inputs["input_ids"]).astype(np.int64)
            
            if "pixel_values" in expected_inputs:
                dummy_shape = [len(chunk_texts), actual_vision_shape[1], actual_vision_shape[2], actual_vision_shape[3]]
                onnx_inputs["pixel_values"] = np.zeros(dummy_shape, dtype=np.float32)

            # 🚀 KLASYCZNE WYWOŁANIE: Odpytujemy stabilnie jeden, konkretny węzeł tekstowy
            raw_outputs = text_session.run([target_output], onnx_inputs)
            text_features = raw_outputs[0]
            if text_features.ndim != 2 or text_features.shape[0] != len(chunk_texts):
                raise ValueError(
                    f"Selected text output '{target_output}' has invalid shape {text_features.shape}; "
                    f"expected ({len(chunk_texts)}, embedding_dim)."
                )

            # Normalizacja L2 (Krytyczna do wyszukiwania cosinusowego)
            norms = np.linalg.norm(text_features, axis=-1, keepdims=True)
            normalized_features = text_features / np.where(norms == 0, 1e-12, norms)
            
            all_features.append(normalized_features)

        return np.concatenate(all_features, axis=0).astype(np.float32)

    except Exception as e:
        print(f"❌ [KRYTYCZNY BŁĄD AI] Awaria w embed_texts_batch: {e}")
        raise e
    
def update_embeddings_from_db(db_path: str, tokenizer, text_session) -> str:
    """
    Ujednolicona, pancerne pętla synchronizacji.
    Sprowadza wszystkie ścieżki (z bazy, FAISS i AI) do pełnego stanu bezwzględnego (Absolute Lowercase),
    eliminując błędy mapowania Windowsa raz na zawsze.
    """
    print("\n=== SYSTEM SYNCHRONIZACJI MULTIMODALNEJ (Pełna Normalizacja Ścieżek) ===")
    start_time = time.time()
    CHECKPOINT_SIZE = 1024

    # 1. Odczyt i konwersja istniejącego cache FAISS do pełnych ścieżek lowercase
    index, image_paths_list = load_index_and_paths()
    if image_paths_list is None:
        image_paths_list = []
    
    normalized_existing_faiss = set()
    for p in image_paths_list:
        try:
            p_obj = Path(p)
            if not p_obj.is_absolute():
                p_obj = IMAGES_DIR / p_obj
            normalized_existing_faiss.add(str(p_obj.resolve()).lower())
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    image_processed_count = 0
    text_processed_count = 0

    try:
        # =====================================================================
        # FAZA 1: PRZETWARZANIE OBRAZÓW
        # =====================================================================
        cursor.execute("""
            SELECT id, jpg_path, image_embedding_exists, image_vector
            FROM models
            WHERE jpg_path IS NOT NULL
        """)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return "Baza danych SQLite jest pusta."

        path_groups = {}
        images_to_embed = set() # Set zapobiega dublowaniu identycznych ścieżek systemowych
        
        for row in rows:
            rel_path = row["jpg_path"].strip()
            if not rel_path:
                continue
                
            p_rel = Path(rel_path)
            if not p_rel.suffix:
                p_rel = p_rel.with_suffix(".jpg")

            # 🚀 KLUCZ: Tworzymy pełną, bezwzględną ścieżkę systemową dla Windowsa
            if p_rel.is_absolute():
                full_path = p_rel.resolve()
            else:
                full_path = (IMAGES_DIR / p_rel).resolve()

            if full_path.exists():
                # Jako klucza używamy absolutnego, rozwiązanego stringu lowercase
                norm_key = str(full_path).lower()
                
                if norm_key not in path_groups:
                    path_groups[norm_key] = []
                path_groups[norm_key].append({"id": row["id"], "flag": row["image_embedding_exists"]})
                
                needs_db_vector = not row["image_vector"]
                if row["image_vector"] and row["image_embedding_exists"] == 0:
                    cursor.execute("UPDATE models SET image_embedding_exists = 1 WHERE id = ?", (row["id"],))

                if needs_db_vector:
                    images_to_embed.add(str(full_path))

        images_to_embed = list(images_to_embed)
        total_images = len(images_to_embed)
        
        if total_images > 0:
            print(f"📸 Wykryto {total_images} nowych unikalnych obrazów do przetworzenia.")
            for i in range(0, total_images, CHECKPOINT_SIZE):
                chunk_paths = images_to_embed[i : i + CHECKPOINT_SIZE]
                
                # chunk_paths są już pełnymi ścieżkami absolutnymi, nie musimy ich dotykać
                new_img_embeddings, valid_absolute_paths = embed_images_batch(chunk_paths)
                
                if new_img_embeddings.shape[0] > 0 and valid_absolute_paths:
                    if index is None:
                        index = faiss.IndexFlatIP(new_img_embeddings.shape[1])

                    faiss_embeddings_to_add = []

                    for idx, abs_path_str in enumerate(valid_absolute_paths):
                        # Konwertujemy ścieżkę zwróconą z AI na nasz wspólny mianownik (abs lowercase)
                        lookup_key = str(Path(abs_path_str).resolve()).lower()
                        
                        if lookup_key not in normalized_existing_faiss:
                            try:
                                rel_path_faiss = str(Path(abs_path_str).relative_to(IMAGES_DIR))
                            except ValueError:
                                rel_path_faiss = abs_path_str

                            image_paths_list.append(rel_path_faiss)
                            normalized_existing_faiss.add(lookup_key)
                            faiss_embeddings_to_add.append(new_img_embeddings[idx])
                        
                        # Mapowanie i aktualizacja rekordów w SQLite
                        vector_json = json.dumps(new_img_embeddings[idx].tolist())
                        associated_models = path_groups.get(lookup_key, [])
                        
                        for m in associated_models:
                            cursor.execute("""
                                UPDATE models 
                                SET image_vector = ?, image_embedding_exists = 1 
                                WHERE id = ?
                            """, (vector_json, m["id"]))
                            image_processed_count += 1

                    if faiss_embeddings_to_add:
                        index.add(np.asarray(faiss_embeddings_to_add, dtype=np.float32))
                
                # Zapis i twardy commit paczki na dysk
                if index is not None:
                    faiss.write_index(index, str(FAISS_INDEX_PATH))
                with open(IMAGE_PATHS_LIST, "w", encoding="utf-8") as f:
                    json.dump(image_paths_list, f, ensure_ascii=False, indent=2)
                conn.commit()
                print(f"🔒 [Checkpoint Obrazów] Zapisano partię na dysku. Łącznie zaktualizowano: {image_processed_count} modeli CAD.")

        conn.commit()

        # =====================================================================
        # FAZA 2: PRZETWARZANIE TEKSTÓW (Bez żadnego śladu po 'category'!)
        # =====================================================================
        cursor.execute("""
            SELECT id, name, manufacturer, opis_produktu, grupa, typ, typ_standardowy
            FROM models 
            WHERE text_embedding_exists = 0 OR text_vector IS NULL
        """)
        text_rows = cursor.fetchall()

        total_texts = len(text_rows)
        if total_texts > 0:
            print(f"📝 Przetwarzanie {total_texts} opisów tekstowych przez DirectML...")
            for i in range(0, total_texts, CHECKPOINT_SIZE):
                chunk_rows = text_rows[i : i + CHECKPOINT_SIZE]
                
                prompts_chunk = []
                for r in chunk_rows:
                    p_text = f"Produkt: {r['name'] or ''}. Producent: {r['manufacturer'] or ''}. Opis: {r['opis_produktu'] or ''}. Grupa: {r['grupa'] or ''}. Typ: {r['typ'] or ''}. Typ standardowy: {r['typ_standardowy'] or ''}."
                    prompts_chunk.append(p_text.strip())
                
                new_text_embeddings = embed_texts_batch(prompts_chunk, tokenizer, text_session) 
                
                for idx, r in enumerate(chunk_rows):
                    text_vector_json = json.dumps(new_text_embeddings[idx].tolist())
                    cursor.execute("""
                        UPDATE models 
                        SET text_vector = ?, text_embedding_exists = 1 
                        WHERE id = ?
                    """, (text_vector_json, r["id"]))
                    text_processed_count += 1
                
                conn.commit()
                print(f"🔒 [Zapis tekstów] Postęp: {i + len(chunk_rows)} / {total_texts}")

        duration = time.time() - start_time
        return f"🏁 Sukces! Zapisano w bazie obrazów: {image_processed_count}, tekstów: {text_processed_count} (Czas: {duration:.2f}s)."

    except Exception as e:
        print(f"❌ Krytyczny błąd podczas pracy skryptu: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
