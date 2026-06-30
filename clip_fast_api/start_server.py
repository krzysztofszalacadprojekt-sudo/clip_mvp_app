import argparse
import socket
import sys
import uvicorn
import multiprocessing
from pathlib import Path

# Importujemy aplikację oraz ścieżkę do głównego katalogu CAD z Twojej konfiguracji
from app.server import app
from app.config import CPP_APP_DIR  

def find_first_free_port(start_port: int, max_scans: int = 50) -> int:
    """Scans ports sequentially starting from start_port to find an available one."""
    for port in range(start_port, start_port + max_scans):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                print(f"⚠️ Port {port} is busy, scanning next...")
                continue
    raise IOError(f"❌ Critical: No free ports found between {start_port} and {start_port + max_scans}")


def update_cpp_config(cpp_dir: Path, port: int):
    """Safely updates or appends the base_url inside the C++ config.txt file."""
    config_file = cpp_dir / "config.txt"
    new_line = f"base_url=http://127.0.0.1:{port}\n"
    
    if not config_file.exists():
        # Jeśli plik z jakiegoś powodu nie istnieje, tworzymy go z naszą linią
        config_file.write_text(new_line, encoding="utf-8")
        print(f"📝 Created new C++ config file with: {new_line.strip()}")
        return

    # Czytamy obecną zawartość pliku line-by-line
    lines = config_file.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = False

    for i, line in enumerate(lines):
        # Szukamy linii zaczynającej się od base_url= (ignorując spacje)
        if line.strip().startswith("base_url="):
            lines[i] = new_line
            updated = True
            break

    # Jeśli w pliku nie było jeszcze klucza base_url=, dopisujemy go na końcu
    if not updated:
        lines.append("\n" + new_line)

    # Zapisujemy zmodyfikowaną zawartość z powrotem na dysk
    config_file.write_text("".join(lines), encoding="utf-8")
    print(f"🚀 Updated C++ config.txt successfully with: {new_line.strip()}")


if __name__ == "__main__":
    # Required to prevent recursive spawning in compiled PyInstaller executables
    multiprocessing.freeze_support()

    # 1. Parser argumentów startowych
    parser = argparse.ArgumentParser(description="Multimodal Search Backend Server")
    parser.add_argument("--port", type=int, default=8000, help="Base port number to start scanning from")
    args = parser.parse_args()

    # 2. 🔒 Bezpieczny host - tylko połączenia lokalne
    secure_host = "127.0.0.1"

    try:
        # 3. Szukanie wolnego portu
        final_port = find_first_free_port(args.port)

        # 4. 🫱‍🫲 HANDSHAKE: Automatyczna aktualizacja pliku config.txt dla C++
        # Używamy wyliczonej w config.py ścieżki CPP_APP_DIR
        update_cpp_config(CPP_APP_DIR, final_port)

        print(f"\n📡 SERVER_ADDRESS_READY -> http://{secure_host}:{final_port}\n")

        # 5. Uruchomienie serwera
        uvicorn.run(app, host=secure_host, port=final_port, log_level="info")

    except Exception as e:
        print(f"❌ FATAL: Backend server failed to initialize: {e}")
        sys.exit(1)