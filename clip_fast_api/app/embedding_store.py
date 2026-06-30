import faiss
import numpy as np
import json
from pathlib import Path
from .config import IMAGES_DIR, FAISS_INDEX_PATH, IMAGE_PATHS_LIST, IMAGES_DIR
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

    normalized_paths = [p.replace("\\", "/").strip() for p in image_paths]

    return index, normalized_paths

def search_similar_images(query_embedding, index, image_paths, top_k=5):
    """
    Bezpieczna funkcja przeszukiwania indeksu FAISS.
    Odporna na błędy synchronizacji, puste wyniki oraz znikające pliki na dysku.
    """
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # 🚀 ROZWIĄZANIE: Pobieramy lekki zapas z FAISS na wypadek, gdyby jakiś plik został usunięty
    search_k = max(top_k + 5, top_k * 2)
    distances, indices = index.search(query_embedding, search_k)
    
    results = []
    if len(indices) == 0 or len(indices[0]) == 0:
        return results
        
    # 🚀 OPTYMALIZACJA: Elegancka iteracja po parach (indeks pętli, wartość z FAISS)
    for i, idx in enumerate(indices[0]):
        # FAISS zwraca -1, jeśli brak mu rekordów w indeksie
        if idx == -1:
            continue
            
        # Defensywne sprawdzenie granic tablicy
        if 0 <= idx < len(image_paths):
            path_to_add = IMAGES_DIR / image_paths[idx]
            
            if path_to_add.exists():
                results.append({
                    "path": str(path_to_add),
                    "distance": float(distances[0][i])
                })
                
                # 🚀 KLUCZOWE: Gdy uzbieramy dokładnie tyle sprawnych plików, ile chciał użytkownik – kończymy
                if len(results) == top_k:
                    break
        else:
            print(f"⚠️ [Defensive Guard] FAISS zwrócił indeks {idx}, ale image_paths ma tylko {len(image_paths)} elementów! Pomijam.")
                
    return results

def update_embeddings_from_db(db_path: str) -> str:
    """
    Lekka, produkcyjna pętla uzgadniania stanu dla ścieżek względnych.
    Gwarantuje relację 1-do-wielu i posiada pancerne zabezpieczenie przed rozjazdem kolejności.
    """
    print("\n=== SYSTEM SYNCHRONIZACJI PRODUKCYJNEJ (Ścieżki Względne) ===")
    start_check = time.time()
    
    # 💡 PODPOWIEDŹ: Jeśli Twoja karta ma mało VRAM, możesz zmniejszyć checkpoint do 256 lub 512
    CHECKPOINT_SIZE = 512 

    # 1. Odczyt aktualnego stanu relatywnego z dysku
    index, image_paths_list = load_index_and_paths()
    if image_paths_list is None:
        image_paths_list = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    path_groups = {}
    images_to_embed = []
    healed_to_exists = 0
    healed_to_missing = 0

    try:
        cursor = conn.cursor()

        # 🚀 POPRAWKA 1: Usunięto sztywny LIMIT 1100, aby pętla mogła przetworzyć całą bazę
        cursor.execute("SELECT id, name, jpg_path, image_embedding_exists FROM models WHERE jpg_path IS NOT NULL ORDER BY id")
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return "Baza danych SQLite jest pusta."

        # KROK A: Grupowanie pod system Windows/Linux
        with conn:
            for row in rows:
                model_id = row["id"]
                db_flag = row["image_embedding_exists"]
                db_path_str = row["jpg_path"].strip()

                if not db_path_str:
                    continue

                p_rel = Path(db_path_str)
                if not p_rel.suffix:
                    p_rel = p_rel.with_suffix(".jpg")
                    db_path_str = str(p_rel)

                full_path = Path(db_path_str)
                if not full_path.is_absolute():
                    full_path = IMAGES_DIR / db_path_str

                if full_path.exists():
                    try:
                        rel_path_str = str(full_path.relative_to(IMAGES_DIR))
                        norm_key = rel_path_str.replace('\\', '/').lower()

                        if norm_key not in path_groups:
                            path_groups[norm_key] = []
                        path_groups[norm_key].append({"id": model_id, "flag": db_flag})
                    except ValueError:
                        print(f"⚠️ Path {db_path_str} is not within the configured IMAGES_DIR. Skipping.")

        normalized_existing = {p.replace('\\', '/').lower() for p in image_paths_list}

        # KROK B: Uzgadnianie flag i kolejka
        with conn:
            for norm_key, models in path_groups.items():
                is_in_faiss = norm_key in normalized_existing

                for m in models:
                    if is_in_faiss and m["flag"] == 0:
                        cursor.execute("UPDATE models SET image_embedding_exists = 1 WHERE id = ?", (m["id"],))
                        healed_to_exists += 1
                    elif not is_in_faiss and m["flag"] == 1:
                        cursor.execute("UPDATE models SET image_embedding_exists = 0 WHERE id = ?", (m["id"],))
                        healed_to_missing += 1

                if not is_in_faiss:
                    images_to_embed.append(norm_key)

        print(f"⏱️ Weryfikacja spójności unikalnych zasobów zakończona w {time.time() - start_check:.4f}s.")
        print(f"📊 Statystyki struktury: Wykryto {len(path_groups)} unikalnych zdjęć dla {len(rows)} modeli bazy.")

        if healed_to_exists > 0 or healed_to_missing > 0:
            print(f"⚡ [Self-Healing] Dostrojono flagi: oznaczono jako istniejące: {healed_to_exists}, zresetowano: {healed_to_missing}")

        total_images_to_process = len(images_to_embed)
        if total_images_to_process == 0:
            return "Wszystkie systemy są w harmonii. Indeks FAISS (Względny) i baza SQLite są idealnie zsynchronizowane."

        print(f"🚀 Do faktycznego przetworzenia przez SigLIP 2 pozostało: {total_images_to_process} UNIKALNYCH obrazów.")
        print(f"💾 Punkty kontrolne (Checkpoints) zabezpieczą dysk co {CHECKPOINT_SIZE} zdjęć.")

        checkpoint_count = 0
        for i in range(0, total_images_to_process, CHECKPOINT_SIZE):
            chunk_paths = images_to_embed[i : i + CHECKPOINT_SIZE]
            current_block_num = (i // CHECKPOINT_SIZE) + 1
            total_blocks = (total_images_to_process + CHECKPOINT_SIZE - 1) // CHECKPOINT_SIZE

            print(f"\n📦 [Blok {current_block_num}/{total_blocks}] Przetwarzanie partii {len(chunk_paths)} unikalnych zdjęć...")

            absolute_chunk_paths = [str(IMAGES_DIR / p) for p in chunk_paths]

            # Inferencja masowa na GPU
            new_img_embeddings, valid_absolute_paths = embed_images_batch(absolute_chunk_paths)

            if new_img_embeddings.shape[0] == 0 or not valid_absolute_paths:
                continue

            # 🚀 POPRAWKA 2: PANCERNY GUARD-RAIL GEOMETRII
            # Jeśli biblioteka pod spodem zgubiła asymetrię między wektorami a ścieżkami, natychmiast zatrzymujemy proces,
            # zamiast pozwolić na ciche uszkodzenie bazy danych.
            if new_img_embeddings.shape[0] != len(valid_absolute_paths):
                raise RuntimeError(
                    f"🛑 [KRYTYCZNY BŁĄD ASYMETRII MOCKA] Funkcja embed_images_batch zwróciła "
                    f"{new_img_embeddings.shape[0]} embeddingów, ale {len(valid_absolute_paths)} ścieżek! "
                    f"Zablokowano zapis checkpointu przed uszkodzeniem indeksu FAISS."
                )

            valid_relative_paths = [str(Path(p).relative_to(IMAGES_DIR)).replace('\\', '/') for p in valid_absolute_paths]

            models_flags_updates = []
            embeddings_upserts = []

            for idx, rel_path in enumerate(valid_relative_paths):
                vector_bytes = new_img_embeddings[idx].astype(np.float32).tobytes()
                lookup_key = rel_path.lower()
                associated_models = path_groups.get(lookup_key, [])

                for m in associated_models:
                    models_flags_updates.append((m["id"],))
                    embeddings_upserts.append((m["id"], vector_bytes))

            # Aktualizacja struktur pamięciowych FAISS
            if index is None:
                index = faiss.IndexFlatIP(new_img_embeddings.shape[1])

            index.add(new_img_embeddings)
            image_paths_list.extend(valid_relative_paths)

            # Atomowy zapis plików indeksu FAISS na dysk twardy
            tmp_index = str(FAISS_INDEX_PATH) + ".tmp"
            faiss.write_index(index, tmp_index)
            os.replace(tmp_index, str(FAISS_INDEX_PATH))

            tmp_json = str(IMAGE_PATHS_LIST) + ".tmp"
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(image_paths_list, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json, IMAGE_PATHS_LIST)

            with conn:
                if models_flags_updates:
                    cursor.executemany("UPDATE models SET image_embedding_exists = 1 WHERE id = ?", models_flags_updates)
                if embeddings_upserts:
                    cursor.executemany("""
                        INSERT INTO model_embeddings (model_id, image_vector)
                        VALUES (?, ?)
                        ON CONFLICT(model_id) DO UPDATE SET image_vector = excluded.image_vector
                    """, embeddings_upserts)

            checkpoint_count += len(valid_relative_paths)
            print(f"🔒 [Checkpoint Zablokowany] Postęp zapisu unikalnych: {i + len(chunk_paths)} / {total_images_to_process}")

        return f"Synchronizacja udana. Przetworzono unikalnych obrazów: {checkpoint_count}."

    except sqlite3.Error as e:
        print(f"❌ Krytyczny błąd bazy danych podczas pętli produkcyjnej V5: {e}")
        raise
    finally:
        conn.close()