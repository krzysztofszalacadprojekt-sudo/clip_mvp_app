#define NOMINMAX
#include <iostream>
#include <string>
#include <curl/curl.h>
#include <fstream>
#include <map>
#include <sstream>
#include <limits>
#include <cstdlib>

std::ofstream logFile;

void menu() {
    std::cout << "\n==== Menu ====\n";
    std::cout << "1. Update embeddings\n";
    std::cout << "2. Search by text\n";
    std::cout << "3. Search by image\n";
    std::cout << "0. Exit\n";
    std::cout << "Choose an option: ";
}

void process_and_display_results(const std::string& raw_json_response) {
    std::cout << "\n📂 Processing search results using system commands...\n";
    
    // 1. Czyszczenie i tworzenie folderu za pomocą natywnych poleceń Windows (CMD)
    // /s /q oznacza usuwanie folderu wraz z zawartością bez pytania użytkownika o zgodę
    // 2>nul wycisza błędy systemu, jeśli folder jeszcze nie istnieje
    std::system("rmdir /s /q search_results 2>nul");
    std::system("mkdir search_results 2>nul");

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

            // 2. Wyciąganie samej nazwy pliku (np. krzeslo.jpg) za pomocą czystego std::string
            size_t last_slash = clean_path.find_last_of("\\/");
            std::string filename = (last_slash == std::string::npos) ? clean_path : clean_path.substr(last_slash + 1);

            // 3. Budowanie systemowego polecenia 'copy' dla Windowsa
            // Bierzemy ścieżki w cudzysłowy \"...\", aby spacje w nazwach folderów (np. CAD Projekt) nie rozbiły komendy
            std::string copy_cmd = "copy \"" + clean_path + "\" \"search_results\\" + std::to_string(file_counter) + "_" + filename + "\" >nul 2>&1";
            
            // Wykonanie kopiowania przez powłokę systemową
            int return_code = std::system(copy_cmd.c_str());
            
            if (return_code == 0) {
                std::cout << "  [+] Match found! Copied: " << filename << "\n";
                file_counter++;
            } else {
                // Kod różny od 0 oznacza zazwyczaj, że plik fizycznie nie istnieje pod wskazaną ścieżką lokalną
                std::cout << "  [-] Could not copy (File not found locally): " << filename << "\n";
            }
        }
        pos = end_quote + 1;
    }

    // 4. Jeśli pliki zostały skopiowane, otwieramy folder w Eksploratorze Windows
    if (file_counter > 1) {
        std::cout << "🎉 Success! Opening 'search_results' folder...\n";
        std::system("start explorer search_results");
    } else {
        std::cout << "⚠️ No valid local image files could be copied from the server response.\n";
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
        std::cout << "❌ Request failed (" << url << "): " << curl_easy_strerror(res) << "\n";
    } else {
        std::cout << "✅ Response from " << url << ":\n" << response << "\n";
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
        std::cout << "❌ Request failed (" << url << "): " << curl_easy_strerror(res) << "\n";
    } else {
        std::cout << "✅ Response from " << url << ":\n" << response << "\n";
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
        std::cout << "❌ Error: Cannot find image file '" << file_path << "' to upload.\n";
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
        std::cout << "❌ Request failed (" << url << "): " << curl_easy_strerror(res) << "\n";
    } else {
        std::cout << "✅ Response from " << url << ":\n" << response << "\n";
        if (logFile.is_open()) {
            logFile << "\n==== RESPONSE ====\n";
            logFile << response << "\n";
        }
    }
    curl_mime_free(form);
    return response;
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
                std::cout << "🚀 Updating embeddings...\n";
                std::string json_payload = build_directories_json(directories);
                send_json_post(curl, base_url + "/update-embeddings", json_payload);
                std::cout << "Done!\n";
                break;
            }

            case 2: {
                std::string search_query;
                std::cout << "🔍 Enter text search query (e.g., modern oak chair): ";
                std::getline(std::cin, search_query);
                
                std::cout << "Searching databases for: '" << search_query << "'...\n";
                std::string response = send_json_post(curl, base_url + "/find-similar-images-by-text", "{\"text\": \"" + search_query + "\", \"top_k\": " + top_k + "}");

                process_and_display_results(response);
                break;
            }

            case 3: {
                // 📂 Pobieramy ścieżkę bezpośrednio z pliku konfiguracyjnego
                std::string image_input = config["target_image"];
                
                if (image_input.empty()) {
                    std::cout << "⚠️ Error: 'target_image' is not defined in config.txt!\n";
                    break;
                }
                
                std::cout << "🖼️ Executing visual lookup for asset from config: '" << image_input << "'...\n";
                
                // Wysyłamy zapytanie do serwera FastAPI
                std::string response = send_multipart_post(curl, base_url + "/find-similar-images-by-image", image_input, top_k);

                // Kopiujemy i otwieramy folder z wynikami
                process_and_display_results(response);
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