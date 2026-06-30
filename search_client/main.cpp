#define NOMINMAX
#include <iostream>
#include <string>
#include <curl/curl.h>
#include <fstream>
#include <map>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <windows.h> // Potrzebne do MultiByteToWideChar
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;
std::ofstream logFile;

void menu() {
    std::cout << "\n==== Menu ====\n";
    std::cout << "1. Update embeddings\n";
    std::cout << "2. Search by text\n";
    std::cout << "3. Search by image\n";
    std::cout << "4. Delete a specific model by file path\n";
    std::cout << "5. Delete a random model\n";
    std::cout << "6. Add a new model\n";
    std::cout << "7. Rebuild index\n";
    std::cout << "0. Exit\n";
    std::cout << "Choose an option: ";
}

std::string safe_json_string(const std::string& input) {
    printf("Debug: Original input string: '%s'\n", input.c_str());
    std::string result;
    for (unsigned char c : input) {
        // 1. Wycinamy niebezpieczne znaki kontrolne ASCII (0-31) oraz bajt zerowy
        if (c < 32) continue; 
        
        // 2. Eskapujemy cudzysłów (bo on może rozbić strukturę JSON)
        if (c == '"') {
            result += "\\\""; 
        } 
        // 3. 🚀 KOMPROMIS: Zamiast eskapować backslash, zamieniamy go na bezpieczną spację
        else if (c == '\\') {
            result += " "; // Bezpieczna separacja słów
        } 
        else {
            result += c;
        }
    }
    return result;
}

std::string console_to_utf8(const std::string& input) {
    if (input.empty()) return "";
    
    // 1. Pobieramy aktualną stronę kodową wejścia terminala klienta (np. 852)
    UINT current_cp = GetConsoleCP();
    
    // Krok A: Konwersja lokalnego kodowania konsoli -> UTF-16 (wstring)
    int wlen = MultiByteToWideChar(current_cp, 0, input.c_str(), (int)input.length(), NULL, 0);
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(current_cp, 0, input.c_str(), (int)input.length(), &wstr[0], wlen);
    
    // Krok B: Konwersja UTF-16 (wstring) -> Czysty, uniwersalny UTF-8 (string)
    int u8len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.length(), NULL, 0, NULL, NULL);
    std::string u8str(u8len, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.length(), &u8str[0], u8len, NULL, NULL);
    
    return u8str;
}

// Pomocnicza funkcja konwertująca UTF-8 string na Windowsowy wstring
std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

void process_and_display_results(const std::string& raw_json_response) {
    std::cout << "\nProcessing search results using native C++ filesystem...\n";
    
    // Czyszczenie i tworzenie folderu w bezpieczny sposób przez bibliotekę C++
    fs::path target_dir("search_results");
    fs::remove_all(target_dir);
    fs::create_directories(target_dir);

    std::string search_key = "\"path\":";
    size_t pos = 0;
    int file_counter = 1;

    while ((pos = raw_json_response.find(search_key, pos)) != std::string::npos) {
        pos += search_key.length();
        
        size_t start_quote = raw_json_response.find("\"", pos);
        size_t end_quote = raw_json_response.find("\"", start_quote + 1);
        
        if (start_quote != std::string::npos && end_quote != std::string::npos) {
            std::string source_path = raw_json_response.substr(start_quote + 1, end_quote - start_quote - 1);
            
            // Naprawa podwójnych backslashy (\\ -> \) z JSON-a
            std::string clean_path = "";
            for (size_t i = 0; i < source_path.length(); ++i) {
                if (source_path[i] == '\\' && i + 1 < source_path.length() && source_path[i+1] == '\\') {
                    clean_path += '\\';
                    i++;
                } else {
                    clean_path += source_path[i];
                }
            }
            if (clean_path.empty()) clean_path = source_path;

            // Konwersja na bezpieczne ścieżki Windows (Wide-char) z pełną obsługą polskich liter
            std::wstring w_clean_path = utf8_to_wstring(clean_path);
            fs::path src_file(w_clean_path);

            if (fs::exists(src_file)) {
                std::wstring w_filename = src_file.filename().wstring();
                std::wstring w_dest_name = std::to_wstring(file_counter) + L"_" + w_filename;
                fs::path dest_file = target_dir / w_dest_name;

                std::error_code ec;
                fs::copy_file(src_file, dest_file, fs::copy_options::overwrite_existing, ec);

                if (!ec) {
                    std::wcout << L"  [+] Match found! Copied: " << w_filename << L"\n";
                    file_counter++;
                } else {
                    std::cout << "  [-] Copy failed (System file system error)\n";
                }
            } else {
                std::wcout << L"  [-] Could not find file locally: " << src_file.filename().wstring() << L"\n";
            }
        }
        pos = end_quote + 1;
    }

    if (file_counter > 1) {
        std::cout << "Success! Opening 'search_results' folder...\n";
        std::system("start explorer search_results");
    } else {
        std::wcout << L"No valid local image files could be copied from the server response.\n";
    }
}

std::map<std::string, std::string> load_config(const std::string& filename) {
    std::map<std::string, std::string> config;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open config file!\n";
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        config[key] = value;
    }

    return config;
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s)
{
    size_t totalSize = size * nmemb;
    std::string data((char*)contents, totalSize);

    s->append(data);

    if (logFile.is_open()) {
        logFile << data << std::endl;
    }

    return totalSize;
}

std::string escape_json_string(const std::string& input)
{
    std::string output;
    output.reserve(input.size());

    for (char c : input)
    {
        if (c == '\\')
        {
            output += "\\\\";
        }
        else if (c == '"')
        {
            output += "\\\"";
        }
        else
        {
            output += c;
        }
    }

    return output;
}

std::string build_directories_json(const std::string& dirs)
{
    std::stringstream ss(dirs);
    std::string dir;

    std::string json = "{\"directories\":[";

    bool first = true;

    while (std::getline(ss, dir, ';'))
    {
        if (!first)
            json += ",";

        json += "\"" + escape_json_string(dir) + "\"";

        first = false;
    }

    json += "]}";

    return json;
}

std::string send_json_post(CURL* curl, const std::string& url, const std::string& json_payload) {

    if (logFile.is_open()) {
        logFile << "\n==== REQUEST ====\n";
        logFile << url << "\n";
        logFile << json_payload << "\n";
    }

    curl_easy_reset(curl); 
    std::string response;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cout << "Request failed (" << url << "): " << curl_easy_strerror(res) << "\n";
    } else {
        std::cout << "Response from " << url << ":\n" << response << "\n";
        if (logFile.is_open()) {
            logFile << "\n==== RESPONSE ====\n";
            logFile << response << "\n";
        }
    }
    curl_slist_free_all(headers);
    return response;
}

std::string send_empty_post(CURL* curl, const std::string& url) {
    curl_easy_reset(curl);
    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L); 
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::wcout << L"Request failed (" << utf8_to_wstring(url) << L"): " << utf8_to_wstring(curl_easy_strerror(res)) << L"\n";
    } else {
        std::wcout << L"Response from " << utf8_to_wstring(url) << L":\n" << utf8_to_wstring(response) << L"\n";
    }
    return response;
}

std::string send_multipart_post(CURL* curl, const std::string& url, const std::string& file_path, const std::string& top_k) {
    if (logFile.is_open()) {
        logFile << "\n==== REQUEST (MULTIPART) ====\n";
        logFile << url << "\n";
        logFile << "file: " << file_path << "\n";
        logFile << "top_k: " << top_k << "\n";
    }

    curl_easy_reset(curl);
    std::string response;
    curl_mime *form = curl_mime_init(curl);
    curl_mimepart *field = NULL;

    FILE* f = fopen(file_path.c_str(), "r");
    if (!f) {
        std::wcout << L"Error: Cannot find image file '" << utf8_to_wstring(file_path) << L"' to upload.\n";
        return "";
    }
    fclose(f);

    field = curl_mime_addpart(form);
    curl_mime_name(field, "file");
    curl_mime_filedata(field, file_path.c_str());

    field = curl_mime_addpart(form);
    curl_mime_name(field, "top_k");
    curl_mime_data(field, top_k.c_str(), CURL_ZERO_TERMINATED);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::wcout << L"Request failed (" << utf8_to_wstring(url) << L"): " << utf8_to_wstring(curl_easy_strerror(res)) << L"\n";
    } else {
        std::wcout << L"Response from " << utf8_to_wstring(url) << L":\n" << utf8_to_wstring(response) << L"\n";
        if (logFile.is_open()) {
            logFile << "\n==== RESPONSE ====\n";
            logFile << response << "\n";
        }
    }
    curl_mime_free(form);
    return response;
}

long send_delete_request(CURL* curl, const std::string& url, std::string& response) {
    curl_easy_reset(curl);
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    long response_code = 0;
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        std::wcout << L"Response from " << utf8_to_wstring(url) << L" (HTTP " << response_code << "):\n" << utf8_to_wstring(response) << L"\n";
    } else {
        std::wcout << L"Request failed (" << utf8_to_wstring(url) << L"): " << utf8_to_wstring(curl_easy_strerror(res)) << L"\n";
    }
    return response_code;
}

int main() {

    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();

    logFile.open("output.txt", std::ios::trunc);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file!\n";
        return 1;
    }

    logFile << "Program started\n";

    const std::string config_filename = "config.txt";
    std::ifstream check_config(config_filename);
    if (!check_config.good()) {
        std::ofstream create_config(config_filename);
        if (create_config.is_open()) {
            create_config << R"(base_url=http://localhost:8000
top_k=5
directories=C:/Users/CAD/Desktop/clip_fast_api/images;
target_image=C:/Users/CAD/Desktop/clip_fast_api/images/sofa.jpg
)"; // <--- TUTAJ: Dodaliśmy domyślną ścieżkę do testowego zdjęcia
            create_config.close();
        }
    } else {
        check_config.close();
    }

    int choice = -1;

    while (choice != 0) {
        menu(); // Upewnij się, że w funkcji menu() zmieniłeś opisy na 1, 2, 3, 0
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            choice = -1;
            continue;
        }

        if (choice == 0) {
            std::cout << "Exiting...\n";
            break;
        }

        // Pobieranie infrastruktury z pliku konfiguracyjnego na bieżąco
        auto config = load_config(config_filename);
        std::string base_url = config["base_url"];
        std::string top_k = config["top_k"];
        std::string directories = config["directories"];

        // Czyszczenie bufora strumienia wejściowego przed użyciem std::getline
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch (choice) {
            case 1: {
                std::wcout << L"Updating embeddings...\n";
                std::string json_payload = build_directories_json(directories);
                send_json_post(curl, base_url + "/update-embeddings", json_payload);
                std::wcout << L"Done!\n";
                break;
            }

            case 2: {
                std::string local_input;
                std::cout << "Enter text search query (e.g., modern oak chair): ";
                
                // 1. Pobieramy surowy ciąg znaków z konsoli
                std::getline(std::cin, local_input);
                
                // 🛑 DEFENSYWNA BLOKADA: Sprawdzamy czy użytkownik wpisał cokolwiek poza spacjami
                if (local_input.empty() || std::all_of(local_input.begin(), local_input.end(), [](unsigned char c) { return std::isspace(c); })) {
                    std::cout << "⚠️ Search query cannot be empty!\n";
                    break;
                }
                
                // 2. Tłumaczymy znaki terminala na standard internetowy UTF-8
                std::string search_query = console_to_utf8(local_input);
                
                // 3. 🚀 KLUCZOWE: Sanityzacja i eskapowanie pod JSON (odcinamy " i \ oraz błędy bajtów)
                std::string escaped_query = safe_json_string(search_query);
                
                // Diagnostyka lokalna (używa oryginalnego stringa do ładnego wyświetlenia)
                std::wcout << L"Searching databases for: '" << utf8_to_wstring(search_query) << L"'...\n";
                
                // 4. Bezpieczne i stabilne złożenie payloadu JSON
                // Używamy std::to_string(top_k), aby jawnie i bezpiecznie przekonwertować int na tekst
                std::string json_payload = "{\"text\": \"" + escaped_query + "\", \"top_k\": " + top_k + "}";
                
                // 5. Wysyłka do FastAPI
                std::string response = send_json_post(curl, base_url + "/find-similar-images-by-text", json_payload);

                process_and_display_results(response);
                break;
            }

            case 3: {
                //  Pobieramy ścieżkę bezpośrednio z pliku konfiguracyjnego
                std::string image_input = config["target_image"];
                
                if (image_input.empty()) {
                    std::cout << "Error: 'target_image' is not defined in config.txt!\n";
                    break;
                }
                
                std::cout << "Executing visual lookup for asset from config: '" << image_input << "'...\n";
                
                // Wysyłamy zapytanie do serwera FastAPI
                std::string response = send_multipart_post(curl, base_url + "/find-similar-images-by-image", image_input, top_k);

                // Kopiujemy i otwieramy folder z wynikami
                process_and_display_results(response);
                break;
            }

            case 4: {
                std::string dwx_path;
                std::cout << "Enter the DWX file path of the model to delete: ";
                std::getline(std::cin, dwx_path);

                if (dwx_path.empty()) {
                    std::cout << "⚠️ Path cannot be empty!\n";
                    break;
                }
                
                char* encoded_path = curl_easy_escape(curl, dwx_path.c_str(), dwx_path.length());
                if(encoded_path) {
                    std::string url = base_url + "/model?dwx_path=" + encoded_path;
                    curl_free(encoded_path);
                    
                    std::string response;
                    long status_code = send_delete_request(curl, url, response);
                    
                    if (status_code >= 200 && status_code < 300) {
                        std::cout << "Model deleted successfully. Now rebuilding index...\n";
                        send_empty_post(curl, base_url + "/rebuild-index");
                    } else {
                        std::cout << "Failed to delete model. Server may be offline. Logging for later.\n";
                        std::ofstream deletion_log("deletions.log", std::ios::app);
                        if (deletion_log.is_open()) {
                            deletion_log << "path:" << dwx_path << std::endl;
                            deletion_log.close();
                        }
                    }
                }
                break;
            }
            case 5: {
                std::cout << "Deleting a random model...\n";
                std::string url = base_url + "/models/random";
                std::string response;
                long status_code = send_delete_request(curl, url, response);

                if (status_code >= 200 && status_code < 300) {
                    std::cout << "Model deleted successfully. Now rebuilding index...\n";
                    send_empty_post(curl, base_url + "/rebuild-index");
                } else {
                    std::cout << "Failed to delete model. Server may be offline. Logging for later.\n";
                    std::ofstream deletion_log("deletions.log", std::ios::app);
                    if (deletion_log.is_open()) {
                        // For random, we don't have an ID, so we can't really log it.
                        // We'll log a generic message. The server will have to handle this.
                        deletion_log << "random:1" << std::endl;
                        deletion_log.close();
                    }
                }
                break;
            }
            case 6: {
                std::string model_path, model_name, jpg_path;
                std::cout << "Enter model name: ";
                std::getline(std::cin, model_name);
                std::cout << "Enter model path (DWX): ";
                std::getline(std::cin, model_path);
                std::cout << "Enter image path (JPG): ";
                std::getline(std::cin, jpg_path);

                if (model_name.empty() || model_path.empty() || jpg_path.empty()) {
                    std::cout << "⚠️ Model name, path, and JPG path cannot be empty!\n";
                    break;
                }

                std::string escaped_name = escape_json_string(model_name);
                std::string escaped_path = escape_json_string(model_path);
                std::string escaped_jpg_path = escape_json_string(jpg_path);

                std::string json_payload = "{\"name\": \"" + escaped_name + "\", \"path\": \"" + escaped_path + "\", \"jpg_path\": \"" + escaped_jpg_path + "\"}";
                send_json_post(curl, base_url + "/add-model", json_payload);
                break;
            }
            case 7: {
                std::cout << "Rebuilding index...\n";
                send_empty_post(curl, base_url + "/rebuild-index");
                break;
            }

            default:
                std::cout << "Invalid choice. Try again.\n";
        }
        
        std::cout << "\n----------------------------------------\n";
    }

    if (curl) {
        curl_easy_cleanup(curl);
    }
    logFile.close();
    return 0;
}