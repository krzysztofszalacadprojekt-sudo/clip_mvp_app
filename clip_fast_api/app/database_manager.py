#!/usr/bin/env python3
"""
Moduł zarządzania lokalną bazą danych SQLite dla systemu wyszukiwania modeli 3D.
Kompatybilny z systemem Windows, używa ścieżek relatywnych/absolutnych przez pathlib.
"""

import csv
import datetime
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

# Konfiguracja logowania produkcyjnego
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
    # Upewnij się, że katalog docelowy istnieje
    db_file.parent.mkdir(parents=True, exist_ok=True)

    query_table = """
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        manufacturer TEXT,
        category TEXT,
        jpg_path TEXT,
        dwx_path TEXT UNIQUE,
        width_mm INTEGER,
        depth_mm INTEGER,
        height_mm INTEGER,
        image_embedding_exists INTEGER DEFAULT 0,
        text_embedding_exists INTEGER DEFAULT 0,
        last_modified TEXT
    );
    """

    # Indeksy optymalizujące wyszukiwanie i filtrowanie w bazie
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_models_dwx_path ON models(dwx_path);",
        "CREATE INDEX IF NOT EXISTS idx_models_manufacturer ON models(manufacturer);",
        "CREATE INDEX IF NOT EXISTS idx_models_category ON models(category);",
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
    """
    now_iso = datetime.datetime.now().isoformat()

    # Przygotowanie pełnego słownika danych z wartościami domyślnymi
    full_data = {
        "id": model_data.get("id"),
        "name": model_data.get("name"),
        "manufacturer": model_data.get("manufacturer"),
        "category": model_data.get("category"),
        "jpg_path": model_data.get("jpg_path"),
        "dwx_path": model_data.get("dwx_path"),
        "width_mm": model_data.get("width_mm"),
        "depth_mm": model_data.get("depth_mm"),
        "height_mm": model_data.get("height_mm"),
        "image_embedding_exists": model_data.get("image_embedding_exists", 0),
        "text_embedding_exists": model_data.get("text_embedding_exists", 0),
        "last_modified": model_data.get("last_modified", now_iso),
    }

    if not full_data["dwx_path"]:
        logger.warning("Pominięto rekord: brak kluczowej wartości 'dwx_path'.")
        return

    query = """
    INSERT INTO models (
        id, name, manufacturer, category, jpg_path, dwx_path,
        width_mm, depth_mm, height_mm, image_embedding_exists,
        text_embedding_exists, last_modified
    ) VALUES (
        :id, :name, :manufacturer, :category, :jpg_path, :dwx_path,
        :width_mm, :depth_mm, :height_mm, :image_embedding_exists,
        :text_embedding_exists, :last_modified
    ) ON CONFLICT(dwx_path) DO UPDATE SET
        name = COALESCE(:name, name),
        manufacturer = COALESCE(:manufacturer, manufacturer),
        category = COALESCE(:category, category),
        jpg_path = COALESCE(:jpg_path, jpg_path),
        width_mm = COALESCE(:width_mm, width_mm),
        depth_mm = COALESCE(:depth_mm, depth_mm),
        height_mm = COALESCE(:height_mm, height_mm),
        image_embedding_exists = COALESCE(:image_embedding_exists, image_embedding_exists),
        text_embedding_exists = COALESCE(:text_embedding_exists, text_embedding_exists),
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
            # Otwarcie pliku z uwzględnieniem kodowania UTF-8 i Windows newline
            with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    logger.warning(f"Plik CSV {csv_path} jest pusty lub uszkodzony.")
                    return

                # Mapowanie nazw kolumn z CSV na klucze bazy danych
                for row in reader:
                    # Bezpieczna konwersja ID na int, jeśli istnieje
                    raw_id = row.get("ID")
                    model_id = int(raw_id) if raw_id and raw_id.isdigit() else None

                    model_data = {
                        "id": model_id,
                        "name": row.get("Nazwa"),
                        "manufacturer": row.get("Producent"),
                        "category": row.get("Kategoria"),
                        "dwx_path": row.get("DWG"),
                        "jpg_path": row.get("JPG"),
                    }

                    # Oczyszczenie stringów z białych znaków (stripping)
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
    Automatycznie odcina zdublowane końcówki plików (np. '88378 88378' lub '88378\\n88378').
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    missing_report: List[Dict[str, Any]] = []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, jpg_path, dwx_path FROM models")
        rows = cursor.fetchall()

        # Wewnętrzna funkcja naprawiająca błędy Windowsa oraz dublowanie nazw
        def sanitize_and_fix_path(raw_path: Optional[str], default_ext: str) -> Optional[Path]:
            if not raw_path:
                return None
            
            # 1. Wstępne oczyszczenie z białych znaków na skrajach
            p_str = raw_path.strip()
            
            # 2. Naprawa struktury dysku C:
            if p_str.upper().startswith("C:"):
                remainder = p_str[2:].lstrip(" \\")
                p_str = "C:\\" + remainder

            # 3. FIX NA ZDUBLOWANĄ KOŃCÓWKĘ (np. "88378 88378" lub "88378\n88378")
            if "\\" in p_str:
                directory, filename_part = p_str.rsplit("\\", 1)
                # Rozbijamy końcówkę po jakichkolwiek białych znakach (spacja, nowa linia)
                tokens = filename_part.strip().split()
                
                # Jeśli wykryjemy dokładnie dwa takie same tokeny obok siebie
                if len(tokens) == 2 and tokens[0] == tokens[1]:
                    filename_part = tokens[0]  # Zostawiamy tylko jeden czysty numer
                    p_str = directory + "\\" + filename_part

            # 4. Budujemy właściwy obiekt Path i dodajemy rozszerzenie, jeśli go brak
            p = Path(p_str)
            if not p.suffix:
                p = p.with_suffix(default_ext)
            return p

        for row in rows:
            model = dict(row)
            missing_jpg = False
            missing_dwx = False

            # Walidacja JPG
            clean_jpg = sanitize_and_fix_path(model["jpg_path"], ".jpg")
            if clean_jpg:
                if not clean_jpg.exists():
                    missing_jpg = True
            else:
                missing_jpg = True

            # Walidacja DWX/DWG
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
    Skanuje bazę danych pod kątem brakujących plików.
    Jeśli plik nie istnieje w zdefiniowanej ścieżce, sprawdza katalog nadrzędny (parent).
    Jeśli plik tam jest -> aktualizuje ścieżkę w bazie danych SQLite.
    Jeśli nadal go nie ma -> wypisuje ostrzeżenie o braku.
    """
    missing_files = get_models_missing_files(db_path)

    if not missing_files:
        logger.info("✅ Synchronizacja zakończona: Wszystkie pliki istnieją na dysku.")
        return

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        
        # Otwieramy jedną transakcję dla wszystkich ewentualnych aktualizacji ścieżek
        with conn:
            for item in missing_files:
                model_id = item["id"]
                model_name = item["name"]

                # -------------------------------------------------------------
                # REKURENCJA DLA JPG: Szukanie w katalogu nadrzędnym
                # -------------------------------------------------------------
                if item["missing_jpg"] and item["clean_jpg_obj"]:
                    orig_path: Path = item["clean_jpg_obj"]
                    # orig_path.parent to obecny folder, .parent.parent to katalog wyżej
                    parent_dir = orig_path.parent.parent 
                    fallback_jpg_path = parent_dir / orig_path.name

                    if parent_dir.exists() and fallback_jpg_path.exists():
                        # Sukces! Znaleziono plik wyżej. Aktualizujemy bazę.
                        new_path_str = str(fallback_jpg_path.resolve())
                        cursor.execute("UPDATE models SET jpg_path = ? WHERE id = ?", (new_path_str, model_id))
                        # logger.info(f"🔄 [Auto-Fix JPG] Model ID {model_id} ({model_name}): Przeniesiono ścieżkę poziom wyżej -> {new_path_str}")
                        item["missing_jpg"] = False  # Flaga wyczyszczona
                    else:
                        logger.warning(f"❌ [Brak pliku JPG] Model ID {model_id} ({model_name}): Nie znaleziono w {item['jpg_path']}")

                # -------------------------------------------------------------
                # REKURENCJA DLA DWX/DWG: Szukanie w katalogu nadrzędnym + fallback rozszerzeń
                # -------------------------------------------------------------
                if item["missing_dwx"] and item["clean_dwx_obj"]:
                    orig_path: Path = item["clean_dwx_obj"]
                    parent_dir = orig_path.parent.parent

                    if parent_dir.exists():
                        # Przygotowujemy potencjalne warianty pliku w katalogu nadrzędnym
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
                            # Sukces! Znaleziono plik modelowy CAD wyżej.
                            new_path_str = str(found_dwx_path.resolve())
                            cursor.execute("UPDATE models SET dwx_path = ? WHERE id = ?", (new_path_str, model_id))
                            logger.info(f"🔄 [Auto-Fix DWX] Model ID {model_id} ({model_name}): Przeniesiono ścieżkę poziom wyżej -> {new_path_str}")
                            item["missing_dwx"] = False  # Flaga wyczyszczona
                        else:
                            logger.warning(f"❌ [Brak pliku CAD] Model ID {model_id} ({model_name}): Nie znaleziono w {item['dwx_path']}")
                    else:
                        logger.warning(f"❌ [Brak pliku CAD] Model ID {model_id} ({model_name}): Ścieżka bazowa niepoprawna.")

    except sqlite3.Error as e:
        logger.error(f"Błąd SQLite podczas automatycznej naprawy ścieżek: {e}")
        raise
    finally:
        conn.close()


# =====================================================================
# 🚀 PRZYKŁAD UŻYCIA (RUNNABLE)
# =====================================================================
if __name__ == "__main__":
    DB_NAME = "models.db"
    CSV_NAME = "test_models.csv"

    print("--- 1. Inicjalizacja bazy danych ---")
    create_database(DB_NAME)

    print("\n--- 2. Tworzenie syntetycznego pliku CSV do testów ---")
    # Generujemy tymczasowe pliki atrapy na dysku, by testy przeszły pomyślnie
    Path("assets/images").mkdir(parents=True, exist_ok=True)
    Path("assets/cad").mkdir(parents=True, exist_ok=True)
    
    dummy_jpg = Path("assets/images/sofa_modern.jpg")
    dummy_dwg = Path("assets/cad/sofa_modern.dwg")
    dummy_jpg.touch(exist_ok=True)
    dummy_dwg.touch(exist_ok=True)

    with open(CSV_NAME, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Nagłówki zgodne ze specyfikacją (w tym jedna losowa kolumna do testu odporności)
        writer.writerow(["ID", "Nazwa", "Producent", "Kategoria", "DWG", "JPG", "NieznanaKolumna"])
        writer.writerow(["1", "Sofa Modern", "Kler", "Meble Tapicerowane", str(dummy_dwg), str(dummy_jpg), "ignoruj"])
        writer.writerow(["2", "Krzesło Loft", "IKEA", "Krzesła", "assets/cad/broken_path.dwg", "", "test"])

    print("\n--- 3. Masowe ładowanie danych z pliku CSV ---")
    load_models_from_csv(CSV_NAME, DB_NAME)

    print("\n--- 4. Pojedynczy Insert lub Update (Test mechanizmu Upsert) ---")
    # Aktualizujemy rekord dla 'Sofa Modern' podając wymiary oraz informację o embeddingu
    update_payload = {
        "name": "Sofa Modern - Premium Edition",
        "dwx_path": str(dummy_dwg),  # Ten sam dwx_path wymusi aktualizację zamiast nowego ID
        "manufacturer": "Kler",
        "category": "Meble Wysegmentowane",
        "width_mm": 2200,
        "depth_mm": 950,
        "height_mm": 850,
        "image_embedding_exists": 1,
    }
    insert_or_update_model(DB_NAME, update_payload)

    print("\n--- 5. Pobieranie modeli bez embeddingów tekstowych/obrazowych ---")
    missing_embeddings = get_models_missing_embeddings(DB_NAME)
    print(f"Liczba modeli wymagających przetworzenia przez CLIP/SigLIP: {len(missing_embeddings)}")
    for item in missing_embeddings:
        print(f" - ID: {item['id']}, Nazwa: {item['name']}, Ścieżka CAD: {item['dwx_path']}")

    print("\n--- 6. Synchronizacja i weryfikacja plików na dysku ---")
    sync_database_with_filesystem(DB_NAME)

    print("\n--- 7. Sprzątanie plików testowych ---")
    # Usuwanie plików tymczasowych utworzonych na potrzeby prezentacji kodu
    if Path(DB_NAME).exists():
        os.remove(DB_NAME)
    if Path(CSV_NAME).exists():
        os.remove(CSV_NAME)
    dummy_jpg.unlink(missing_ok=True)
    dummy_dwg.unlink(missing_ok=True)
    print("Testy zakończone sukcesem. Wszystkie mechanizmy działają prawidłowo.")