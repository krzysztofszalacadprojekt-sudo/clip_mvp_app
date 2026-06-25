#!/usr/bin/env python3
"""
Execution script to completely wipe any existing database and build a 
fresh, structurally sound SQLite database directly from the source CSV.
"""

import logging
import os
from pathlib import Path

# Import your updated management functions
from app.database_manager import create_database, load_models_from_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    # 1. Setup paths relative to this script execution location
    BASE_DIR = Path(__file__).resolve().parent
    TARGET_DB = BASE_DIR / "data" / "models.db"
    SOURCE_CSV = BASE_DIR / "data" / "dodatki.csv"

    print("====================================================================")
    print("🧹 PURGING AND REBUILDING VECTOR HYBRID DATABASE FROM SCRATCH")
    print("====================================================================")

    # 2. Defensively drop the old database file to guarantee a zero-error clean slate
    if TARGET_DB.exists():
        try:
            logger.info(f"Removing old database file: {TARGET_DB.resolve()}")
            os.remove(TARGET_DB)
            logger.info("🗑️ Legacy database wiped successfully.")
            for cache_file in [BASE_DIR / "data" / "index.bin", BASE_DIR / "data" / "image_paths.json"]:
                if cache_file.exists():
                    os.remove(cache_file)
                    logger.info(f"🗑️ Usunięto stary cache indeksu: {cache_file.name}")
        except OSError as e:
            logger.error(f"❌ Failed to delete old database file (Is it locked?): {e}")
            print("Please close any open DB Connections or IDE editors and retry.")
            return

    # 3. Create a brand-new database file with the correct schema layout
    try:
        logger.info("Building brand new database schema layout...")
        create_database(str(TARGET_DB))
        
        # 4. Perform a pristine, non-conflicting bulk import from your raw CSV
        logger.info("Beginning fresh CSV catalog data initialization...")
        load_models_from_csv(csv_path=str(SOURCE_CSV), db_path=str(TARGET_DB))
        
        print("\n🎉 SUCCESS! Pristine database initialized flawlessly.")
        print(f"Location: {TARGET_DB.resolve()}")
        print("Both 'image_vector' and 'text_vector' text storage fields are ready.")
        print("====================================================================")

    except Exception as e:
        print(f"\n❌ CRITICAL CRASH: Failed to instantiate database: {e}")
        print("====================================================================")


if __name__ == "__main__":
    main()