#!/usr/bin/env python3
"""
Moduł profesjonalnego zarządzania lokalną bazą danych SQLite dla systemu modeli 3D.
Wprowadza binarny zapis wektorów (BLOB) float32 i separację tabel 1:1 (Score Fusion Ready).
Kompatybilny z systemem Windows, używa pancernej obsługi ścieżek przez pathlib.
"""

import csv
import datetime
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_database(db_path: str) -> None:
    """
    Tworzy zoptymalizowaną strukturę bazy danych SQLite z podziałem na tabelę
    metadanych mebli oraz wydzieloną tabelę ciężkich wektorów binarnych AI (BLOB).
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Główna tabela metadanych (Lekka, ultraszybka dla zapytań z C++)
    query_models_table = """
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        manufacturer TEXT,
        opis_produktu TEXT,
        grupa TEXT,
        typ TEXT,
        typ_standardowy TEXT,
        jpg_path TEXT,
        dwx_path TEXT UNIQUE,
        image_embedding_exists INTEGER DEFAULT 0,
        text_embedding_exists INTEGER DEFAULT 0,
        last_modified TEXT
    );
    """

    # 2. Wydzielona tabela wektorowa (Przechowuje surowe bajty float32 o rozmiarze 3072 bajtów)
    query_embeddings_table = """
    CREATE TABLE IF NOT EXISTS model_embeddings (
        model_id INTEGER PRIMARY KEY,
        image_vector BLOB,   -- Surowy zrzut pamięci tablicy NumPy float32 (768 * 4 bajty)
        text_vector BLOB,    -- Surowy zrzut pamięci tablicy NumPy float32 (768 * 4 bajty)
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
    );
    """

    # Indeksy wydajnościowe dla pól wyszukiwanych przez silnik CAD i API
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_models_dwx_path ON models(dwx_path);",
        "CREATE INDEX IF NOT EXISTS idx_models_manufacturer ON models(manufacturer);",
        "CREATE INDEX IF NOT EXISTS idx_models_img_exists ON models(image_embedding_exists);",
        "CREATE INDEX IF NOT EXISTS idx_models_txt_exists ON models(text_embedding_exists);"
    ]

    conn = sqlite3.connect(str(db_file))
    try:
        # Włączamy kaskadowe usuwanie powiązań (Foreign Key Support)
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            conn.execute(query_models_table)
            conn.execute(query_embeddings_table)
            for index_query in indexes:
                conn.execute(index_query)
        logger.info(f"💾 [SQL OK] Baza danych rozbita na dwie tabele została przygotowana w: {db_file.resolve()}")
    except sqlite3.Error as e:
        logger.error(f"❌ Błąd podczas tworzenia struktur bazy danych: {e}")
        raise
    finally:
        conn.close()


def _execute_upsert(cursor: sqlite3.Cursor, model_data: Dict[str, Any]) -> None:
    """
    Wewnętrzna funkcja realizująca bezpieczne zapytanie UPSERT dla metadanych.
    Zabezpiecza przed nadpisywaniem flag synchronizacji przez ponowny import z CSV.
    """
    now_iso = datetime.datetime.now().isoformat()

    full_data = {
        "name": model_data.get("name"),
        "manufacturer": model_data.get("manufacturer"),
        "opis_produktu": model_data.get("opis_produktu"),
        "grupa": model_data.get("grupa"),
        "typ": model_data.get("typ"),
        "typ_standardowy": model_data.get("typ_standardowy"),
        "jpg_path": model_data.get("jpg_path"),
        "dwx_path": model_data.get("dwx_path"),
        "image_embedding_exists": model_data.get("image_embedding_exists"),
        "text_embedding_exists": model_data.get("text_embedding_exists"),
        "last_modified": model_data.get("last_modified", now_iso),
    }

    if not full_data["dwx_path"]:
        logger.warning("Pominięto rekord: brak kluczowego unikalnego pola 'dwx_path'.")
        return

    query = """
    INSERT INTO models (
        name, manufacturer, opis_produktu, grupa, typ, typ_standardowy,
        jpg_path, dwx_path, image_embedding_exists, text_embedding_exists, last_modified
    ) VALUES (
        :name, :manufacturer, :opis_produktu, :grupa, :typ, :typ_standardowy,
        :jpg_path, :dwx_path,
        COALESCE(:image_embedding_exists, 0), 
        COALESCE(:text_embedding_exists, 0), 
        :last_modified
    ) ON CONFLICT(dwx_path) DO UPDATE SET
        name = COALESCE(:name, name),
        manufacturer = COALESCE(:manufacturer, manufacturer),
        opis_produktu = COALESCE(:opis_produktu, opis_produktu),
        grupa = COALESCE(:grupa, grupa),
        typ = COALESCE(:typ, typ),
        typ_standardowy = COALESCE(:typ_standardowy, typ_standardowy),
        jpg_path = COALESCE(:jpg_path, jpg_path),
        image_embedding_exists = COALESCE(:image_embedding_exists, image_embedding_exists),
        text_embedding_exists = COALESCE(:text_embedding_exists, text_embedding_exists),
        last_modified = :last_modified;
    """
    cursor.execute(query, full_data)


def insert_or_update_model(db_path: str, model_data: Dict[str, Any]) -> None:
    """
    Interfejs API dla aplikacji zewnętrznych (np. wywołanie z serwera FastAPI).
    Dodaje model lub aktualizuje istniejący na podstawie unikalnej ścieżki CAD.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            cursor = conn.cursor()
            _execute_upsert(cursor, model_data)
    except sqlite3.Error as e:
        logger.error(f"❌ Błąd podczas pojedynczej operacji insert_or_update: {e}")
        raise
    finally:
        conn.close()


def load_models_from_csv(csv_path: str, db_path: str) -> None:
    """
    Masowo importuje lub aktualizuje bazę danych z pliku CSV w jednej transakcji ACID.
    Idealnie parsuje i czyści mapowanie kluczy Windowsa oraz polskie znaki UTF-8.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Plik CSV nie istnieje pod wskazanym adresem: {csv_file.resolve()}")
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            cursor = conn.cursor()
            with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    logger.warning(f"Plik CSV {csv_path} jest pusty lub uszkodzony strukturalnie.")
                    return

                for row in reader:
                    model_data = {
                        "name": row.get("Nazwa"),
                        "dwx_path": row.get("DWG"),
                        "jpg_path": row.get("JPG"),
                        "manufacturer": row.get("Producent"),
                        "opis_produktu": row.get("Opis_Produktu"),
                        "grupa": row.get("Grupa"),
                        "typ": row.get("Typ"),
                        "typ_standardowy": row.get("typ_standardowy"),
                    }

                    # Usuwamy zbędne białe znaki ze skrajów tekstu
                    cleaned_data = {
                        k: (v.strip() if isinstance(v, str) else v)
                        for k, v in model_data.items()
                    }

                    _execute_upsert(cursor, cleaned_data)

        logger.info(f"✅ [CSV IMPORT SUCCESS] Pomyślnie zsynchronizowano bazę danych z plikiem: {csv_path}")
    except (sqlite3.Error, OSError) as e:
        logger.error(f"❌ Błąd krytyczny podczas ładowania danych z pliku CSV: {e}")
        raise
    finally:
        conn.close()


def get_models_missing_files(db_path: str) -> List[Dict[str, Any]]:
    """
    Skanuje bazę i weryfikuje fizyczną dostępność plików CAD i JPG na dyskach Windowsa.
    Naprawia anomalie powielonych nazw plików z generatorów ścieżek.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    missing_report: List[Dict[str, Any]] = []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, jpg_path, dwx_path FROM models")
        rows = cursor.fetchall()

        def sanitize_and_fix_path(raw_path: Optional[str], default_ext: str) -> Optional[Path]:
            if not raw_path:
                return None
            
            p_str = raw_path.strip()
            
            if p_str.upper().startswith("C:"):
                remainder = p_str[2:].lstrip(" \\")
                p_str = "C:\\" + remainder

            if "\\" in p_str:
                directory, filename_part = p_str.rsplit("\\", 1)
                tokens = filename_part.strip().split()
                
                if len(tokens) == 2 and tokens[0] == tokens[1]:
                    filename_part = tokens[0]
                    p_str = directory + "\\" + filename_part

            p = Path(p_str)
            if not p.suffix:
                p = p.with_suffix(default_ext)
            return p

        for row in rows:
            model = dict(row)
            missing_jpg = False
            missing_dwx = False

            clean_jpg = sanitize_and_fix_path(model["jpg_path"], ".jpg")
            if clean_jpg:
                if not clean_jpg.exists():
                    missing_jpg = True
            else:
                missing_jpg = True

            clean_dwx = sanitize_and_fix_path(model["dwx_path"], ".dwx")
            if clean_dwx:
                path_as_is = clean_dwx
                path_as_dwx = clean_dwx.with_suffix(".dwx")
                path_as_dwg = clean_dwx.with_suffix(".dwg")

                if not path_as_is.exists() and not path_as_dwx.exists() and not path_as_dwg.exists():
                    missing_dwx = True
            else:
                missing_dwx = True

            if missing_jpg or missing_dwx:
                model["missing_jpg"] = missing_jpg
                model["missing_dwx"] = missing_dwx
                model["clean_jpg_obj"] = clean_jpg
                model["clean_dwx_obj"] = clean_dwx
                missing_report.append(model)

        return missing_report
    except sqlite3.Error as e:
        logger.error(f"❌ Błąd silnika walidacji integralności plików: {e}")
        raise
    finally:
        conn.close()


def sync_database_with_filesystem(db_path: str) -> None:
    """
    Automatyczny system naprawczy (Auto-Heal). Jeśli plik zaginął, skanuje katalog wyżej
    i automatycznie aktualizuje rekord w bazie, jeśli plik tam emigrował.
    """
    missing_files = get_models_missing_files(db_path)

    if not missing_files:
        logger.info("✅ Integralność plików systemowych zachowana. Brak anomalii ścieżek.")
        return

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        with conn:
            for item in missing_files:
                model_id = item["id"]
                model_name = item["name"]

                if item["missing_jpg"] and item["clean_jpg_obj"]:
                    orig_path: Path = item["clean_jpg_obj"]
                    parent_dir = orig_path.parent.parent 
                    fallback_jpg_path = parent_dir / orig_path.name

                    if parent_dir.exists() and fallback_jpg_path.exists():
                        new_path_str = str(fallback_jpg_path.resolve())
                        cursor.execute("UPDATE models SET jpg_path = ? WHERE id = ?", (new_path_str, model_id))
                        item["missing_jpg"] = False
                    else:
                        logger.warning(f"⚠️ [Brak pliku JPG] Model ID {model_id} ({model_name}): Plik usunięty z dysku.")

                if item["missing_dwx"] and item["clean_dwx_obj"]:
                    orig_path: Path = item["clean_dwx_obj"]
                    parent_dir = orig_path.parent.parent

                    if parent_dir.exists():
                        fallback_as_is = parent_dir / orig_path.name
                        fallback_as_dwx = parent_dir / orig_path.with_suffix(".dwx").name
                        fallback_as_dwg = parent_dir / orig_path.with_suffix(".dwg").name

                        found_dwx_path: Optional[Path] = None
                        if fallback_as_is.exists():
                            found_dwx_path = fallback_as_is
                        elif fallback_as_dwx.exists():
                            found_dwx_path = fallback_as_dwx
                        elif fallback_as_dwg.exists():
                            found_dwx_path = fallback_as_dwg

                        if found_dwx_path:
                            new_path_str = str(found_dwx_path.resolve())
                            cursor.execute("UPDATE models SET dwx_path = ? WHERE id = ?", (new_path_str, model_id))
                            logger.info(f"🔄 [Auto-Fix CAD] Model ID {model_id} ({model_name}): Naprawiono relację struktury katalogów -> {new_path_str}")
                            item["missing_dwx"] = False
                        else:
                            logger.warning(f"⚠️ [Brak pliku CAD] Model ID {model_id} ({model_name}): Plik bryły 3D usunięty z dysku.")
    except sqlite3.Error as e:
        logger.error(f"❌ Wyjątek SQLite podczas automatycznego leczenia ścieżek: {e}")
        raise
    finally:
        conn.close()


def delete_model_by_id(db_path: str, model_id: int) -> bool:
    """
    Deletes a model from the database by its ID.
    Thanks to 'ON DELETE CASCADE', associated embeddings are also removed.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            # Check if any row was affected
            if cursor.rowcount > 0:
                logger.info(f"✅ Successfully deleted model with ID: {model_id}")
                return True
            else:
                logger.warning(f"⚠️ Model with ID {model_id} not found.")
                return False
    except sqlite3.Error as e:
        logger.error(f"❌ Error while deleting model with ID {model_id}: {e}")
        return False
    finally:
        conn.close()


def delete_random_model(db_path: str) -> Optional[int]:
    """
    Deletes a random model from the database.
    Returns the ID of the deleted model, or None if the database is empty.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            cursor = conn.cursor()
            # First, get a random model ID
            cursor.execute("SELECT id FROM models ORDER BY RANDOM() LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                model_id = result[0]
                cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
                logger.info(f"✅ Successfully deleted random model with ID: {model_id}")
                return model_id
            else:
                logger.warning("⚠️ Database is empty. No model to delete.")
                return None
    except sqlite3.Error as e:
        logger.error(f"❌ Error while deleting random model: {e}")
        return None
    finally:
        conn.close()


def delete_model_by_dwx_path(db_path: str, dwx_path: str) -> bool:
    """
    Deletes a model from the database by its unique dwx_path.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE dwx_path = ?", (dwx_path,))
            if cursor.rowcount > 0:
                logger.info(f"✅ Successfully deleted model with dwx_path: {dwx_path}")
                return True
            else:
                logger.warning(f"⚠️ Model with dwx_path '{dwx_path}' not found.")
                return False
    except sqlite3.Error as e:
        logger.error(f"❌ Error while deleting model with dwx_path {dwx_path}: {e}")
        return False
    finally:
        conn.close()