#!/usr/bin/env python3
"""
Skrypt automatycznego wdrażania nowej struktury bazodanowej.
Kasuje stare dane, buduje tabele relacyjne, importuje CSV i uruchamia synchronizację AI.
"""

import os
from pathlib import Path
import time

# Importujemy metody z Twoich zaktualizowanych modułów
from app.database_manager import create_database, load_models_from_csv, sync_database_with_filesystem
from app.embedding_store import update_embeddings_from_db
from app import config # Upewnij się, że ścieżka do configu jest prawidłowa

def main():
    print("🧹 [MIGRACJA] Rozpoczynanie procesu budowy bazy danych klasy Enterprise...")
    
    # 1. Definiowanie ścieżek na podstawie Twojej konfiguracji
    DB_PATH = str(config.DB_PATH)
    CSV_PATH = "data/dodatki.csv" # <--- Wpisz tutaj poprawną ścieżkę do swojego pliku CSV
    
    # 2. Usuwanie starej bazy danych, aby zapobiec konfliktom struktur (Conflict Schema)
    if os.path.exists(DB_PATH):
        print(f"🗑️ Usuwanie wykrytej starej bazy danych: {DB_PATH}")
        try:
            os.remove(DB_PATH)
            print("✅ Stara baza została trwale usunięta.")
        except Exception as e:
            print(f"❌ Nie można usunąć pliku bazy (może jest otwarty w DB Browser?): {e}")
            return

    for faiss_file in ["data/faiss_index.bin", "data/image_paths.json"]:
        if os.path.exists(faiss_file):
            os.remove(faiss_file)
            print(f"🗑️ Usunięto stary cache FAISS: {faiss_file}")

    print("\n--- KROK 1: Inicjalizacja nowych tabel binarnych (BLOB) ---")
    start = time.time()
    create_database(DB_PATH)
    
    print("\n--- KROK 2: Masowy import metadanych mebli z pliku CSV ---")
    if os.path.exists(CSV_PATH):
        load_models_from_csv(CSV_PATH, DB_PATH)
    else:
        print(f"⚠️ Ostrzeżenie: Nie znaleziono pliku CSV pod ścieżką '{CSV_PATH}'. Pomijanie importu.")

    print("\n--- KROK 3: Automatyczna weryfikacja i leczenie ścieżek dyskowych (Auto-Heal) ---")
    sync_database_with_filesystem(DB_PATH)

    print("\n--- KROK 4: Uruchomienie pętli AI i masowy zapis wektorów BLOB do SQLite ---")
    # Ponieważ baza jest nowa, funkcja wykryje brak wektorów i zacznie je bezpiecznie pakować do tabeli model_embeddings
    message = update_embeddings_from_db(DB_PATH)
    print(f"\nℹ️ Komunikat z silnika synchronizacji: {message}")
    
    end = time.time()
    print(f"\n🏁 [SUKCES] Nowy plik bazy danych gotowy! Całkowity czas operacji: {end - start:.2f}s.")

if __name__ == "__main__":
    main()