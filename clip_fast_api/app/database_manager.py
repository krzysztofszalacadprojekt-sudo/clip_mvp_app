#!/usr/bin/env python3
"""
Moduł zarządzania lokalną bazą danych SQLite dla systemu wyszukiwania modeli 3D.
Kompatybilny z systemem Windows, używa ścieżek relatywnych/absolutnych przez pathlib.
Wzbogacony o trwały magazyn wektorów (embeddings) dla obrazów i tekstu.
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
    Tworzy bazę danych SQLite oraz tabelę 'models' wraz z niezbędnymi indeksami,
    jeśli one jeszcze nie istnieją.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. ZAKTUALIZOWANA STRUKTURA: Dodano nowe kolumny tekstowe z CSV
    query_table = """
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
        image_vector TEXT,  -- Trwały cache wektora obrazu (Zserializowany JSON string)
        text_vector TEXT,   -- Trwały cache wektora tekstu (Zserializowany JSON string)
        last_modified TEXT
    );
    """

    # Indeksy optymalizujące wyszukiwanie i filtrowanie w bazie
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_models_dwx_path ON models(dwx_path);",
        "CREATE INDEX IF NOT EXISTS idx_models_manufacturer ON models(manufacturer);",
    ]

    conn = sqlite3.connect(str(db_file))
    try:
        with conn:
            conn.execute(query_table)
            for index_query in indexes:
                conn.execute(index_query)
        logger.info(f"Baza danych i indeksy zostały przygotowane w: {db_file.resolve()}")
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas tworzenia bazy danych: {e}")
        raise
    finally:
        conn.close()


def _execute_upsert(cursor: sqlite3.Cursor, model_data: Dict[str, Any]) -> None:
    """
    Wewnętrzna funkcja pomocnicza realizująca zapytanie UPSERT na otwartym kursorze.
    Bezpiecznie dba o to, by nie wyczyścić istniejących wektorów przy braku danych wejściowych.
    """
    now_iso = datetime.datetime.now().isoformat()

    # 2. MAPOWANIE SŁOWNIKA: Przekazujemy nowe kolumny z zachowaniem spójnego snake_case
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
        "image_vector": model_data.get("image_vector"),
        "text_vector": model_data.get("text_vector"),
        "last_modified": model_data.get("last_modified", now_iso),
    }

    if not full_data["dwx_path"]:
        logger.warning("Pominięto rekord: brak kluczowej wartości 'dwx_path'.")
        return

    # 3. ZAKTUALIZOWANA KWERENDA SQL: Dodano obsługę nowych pól w sekcjach INSERT oraz DO UPDATE
    query = """
    INSERT INTO models (
        name, manufacturer, opis_produktu, grupa, typ, typ_standardowy,
        jpg_path, dwx_path, image_embedding_exists,
        text_embedding_exists, image_vector, text_vector, last_modified
    ) VALUES (
        :name, :manufacturer, :opis_produktu, :grupa, :typ, :typ_standardowy,
        :jpg_path, :dwx_path,
        COALESCE(:image_embedding_exists, 0), 
        COALESCE(:text_embedding_exists, 0), 
        :image_vector, :text_vector, :last_modified
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
        image_vector = COALESCE(:image_vector, image_vector),
        text_vector = COALESCE(:text_vector, text_vector),
        last_modified = :last_modified;
    """
    cursor.execute(query, full_data)


def insert_or_update_model(db_path: str, model_data: Dict[str, Any]) -> None:
    """
    Dodaje nowy model lub aktualizuje istniejący na podstawie unikalnego 'dwx_path'.
    """
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            _execute_upsert(cursor, model_data)
    except sqlite3.Error as e:
        logger.error(f"Błąd podczas operacji insert_or_update: {e}")
        raise
    finally:
        conn.close()


def load_models_from_csv(csv_path: str, db_path: str) -> None:
    """
    Wczytuje modele z pliku CSV i masowo (w jednej transakcji) aktualizuje bazę danych.
    Obsługuje brakujące kolumny bez przerywania działania programu.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Plik CSV nie istnieje: {csv_file.resolve()}")
        return

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    logger.warning(f"Plik CSV {csv_path} jest pusty lub uszkodzony.")
                    return

                for row in reader:
                    # 4. ZGODNOŚĆ KLUCZY: Zamieniłem wielkie litery na czysty standard snake_case
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

                    cleaned_data = {
                        k: (v.strip() if isinstance(v, str) else v)
                        for k, v in model_data.items()
                    }

                    _execute_upsert(cursor, cleaned_data)

        logger.info(f"Pomyślnie przetworzono i zaimportowano dane z pliku CSV: {csv_path}")
    except (sqlite3.Error, OSError) as e:
        logger.error(f"Błąd podczas ładowania danych z pliku CSV: {e}")
        raise
    finally:
        conn.close()


def get_models_missing_files(db_path: str) -> List[Dict[str, Any]]:
    """
    Weryfikuje fizyczną obecność plików na dysku (JPG i DWX).
    Automatycznie odcina zdublowane końcówki plików (np. '88378 88378').
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
        logger.error(f"Błąd podczas weryfikacji plików na dysku: {e}")
        raise
    finally:
        conn.close()


def sync_database_with_filesystem(db_path: str) -> None:
    """
    Skanuje bazę danych pod kątem brakujących plików i automatycznie naprawia ścieżki
    w katalogu nadrzędnym (parent directory), jeśli pliki tam istnieją.
    """
    missing_files = get_models_missing_files(db_path)

    if not missing_files:
        logger.info("✅ Synchronizacja zakończona: Wszystkie pliki istnieją na dysku.")
        return

    conn = sqlite3.connect(db_path)
    try:
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
                        logger.warning(f"❌ [Brak pliku JPG] Model ID {model_id} ({model_name}): Nie znaleziono w {item['jpg_path']}")

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
                            logger.info(f"🔄 [Auto-Fix DWX] Model ID {model_id} ({model_name}): Przeniesiono ścieżkę poziom wyżej -> {new_path_str}")
                            item["missing_dwx"] = False
                        else:
                            logger.warning(f"❌ [Brak pliku CAD] Model ID {model_id} ({model_name}): Nie znaleziono w {item['dwx_path']}")
                    else:
                        logger.warning(f"❌ [Brak pliku CAD] Model ID {model_id} ({model_name}): Ścieżka bazowa niepoprawna.")

    except sqlite3.Error as e:
        logger.error(f"Błąd SQLite podczas automatycznej naprawy ścieżek: {e}")
        raise
    finally:
        conn.close()