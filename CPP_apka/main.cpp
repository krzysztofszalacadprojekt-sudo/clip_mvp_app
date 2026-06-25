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
#include <thread>
#include <sqlite3.h>

namespace fs = std::filesystem;
std::ofstream logFile;

void menu() {
    std::cout << "\n==== Menu ====\n";
    std::cout << "1. Update embeddings\n";
    std::cout << "2. Search by text (Hybrid Multimodal)\n"; // 🚀 Zaktualizowany opis opcji
    std::cout << "3. Search by image\n";
    std::cout << "4. Add new model\n";
    std::cout << "5. Delete model manually\n";
    std::cout << "0. Exit\n";
    std::cout << "Choose an option: ";
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s);

void TriggerBackendSync(const std::string& base_url)
{
    std::thread([base_url]() {
        CURL* curl = curl_easy_init();
        if (curl) {
            std::string url = base_url + "/sync";
            std::string response;

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L); 
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK) {
                std::cout << "\n📦 [Background Sync] FAISS rebuild triggered successfully.\n";
            }
            curl_easy_cleanup(curl);
        }
    }).detach();
}

bool DeleteModelFromDB(sqlite3* db, const std::string& dwx_path, const std::string& base_url)
{
    const char* sql = "DELETE FROM models WHERE dwx_path = ?;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "❌ SQL Error (prepare): " << sqlite3_errmsg(db) << "\n";
        return false;
    }

    sqlite3_bind_text(stmt, 1, dwx_path.c_str(), -1, SQLITE_TRANSIENT);

    bool success = false;
    
    if (sqlite3_step(stmt) == SQLITE_DONE) {
        int rowsAffected = sqlite3_changes(db);
        if (rowsAffected > 0) {
            std::cout << "💥 Model removed from SQLite successfully.\n";
            TriggerBackendSync(base_url);
            success = true;
        } else {
            std::cout << "⚠️ No model found with given path: " << dwx_path << "\n";
        }
    } else {
        std::cerr << "❌ SQL Error (execute): " << sqlite3_errmsg(db) << "\n";
    }

    sqlite3_finalize(stmt);
    return success;
}

bool AddNewModelToDB(sqlite3* db, 
                     const std::string& name, 
                     const std::string& manufacturer, 
                     const std::string& dwx_path, 
                     const std::string& jpg_path,
                     const std::string& base_url)
{
    const char* sql = "INSERT OR REPLACE INTO models (name, manufacturer, dwx_path, jpg_path, image_embedding_exists, text_embedding_exists, grupa, typ, typ_standardowy, opis_produktu) "
                      "VALUES (?, ?, ?, ?, 0, 0, '', '', '', '');";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "❌ SQL Error (prepare): " << sqlite3_errmsg(db) << "\n";
        return false;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, manufacturer.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, dwx_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, jpg_path.c_str(), -1, SQLITE_TRANSIENT);

    bool success = false;

    if (sqlite3_step(stmt) == SQLITE_DONE) {
        std::cout << "✨ New model injected into SQLite.\n";
        TriggerBackendSync(base_url);
        success = true;
    } else {
        std::cerr << "❌ SQL Error (execute): " << sqlite3_errmsg(db) << "\n";
    }

    sqlite3_finalize(stmt);
    return success;
}

std::string safe_json_string(const std::string& input) {
    std::string result;
    for (unsigned char c : input) {
        if (c < 32) { result += " "; } 
        else if (c == '"') { result += "\\\""; } 
        else if (c == '\\') { result += " "; } 
        else { result += c; }
    }
    return result;
}

std::string console_to_utf8(const std::string& input) {
    if (input.empty()) return "";
    UINT current_cp = GetConsoleCP();
    
    int wlen = MultiByteToWideChar(current_cp, 0, input.c_str(), (int)input.length(), NULL, 0);
    std::wstring wstr(wlen, 0);
    MultiByteToWideChar(current_cp, 0, input.c_str(), (int)input.length(), &wstr[0], wlen);
    
    int u8len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.length(), NULL, 0, NULL, NULL);
    std::string u8str(u8len, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.length(), &u8str[0], u8len, NULL, NULL);
    
    return u8str;
}

std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

void process_and_display_results(const std::string& raw_json_response) {
    std::cout << "\nProcessing search results using native C++ filesystem...\n";
    
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
    if (logFile.is_open()) { logFile << data << std::endl; }
    return totalSize;
}

std::string escape_json_string(const std::string& input)
{
    std::string output;
    output.reserve(input.size());
    for (char c : input)
    {
        if (c == '\\') { output += "\\\\"; }
        else if (c == '"') { output += "\\\""; }
        else { output += c; }
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
        if (!first) json += ",";
        json += "\"" + escape_json_string(dir) + "\"";
        first = false;
    }

    json += "]}";
    return json;
}

std::string send_json_post(CURL* curl, const std::string& url, const std::string& json_payload) {
    if (logFile.is_open()) {
        logFile << "\n==== REQUEST ====\n" << url << "\n" << json_payload << "\n";
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
        std::cout << "Response received from server.\n";
        if (logFile.is_open()) {
            logFile << "\n==== RESPONSE ====\n" << response << "\n";
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
    return response;
}

std::string send_multipart_post(CURL* curl, const std::string& url, const std::string& file_path, const std::string& top_k) {
    if (logFile.is_open()) {
        logFile << "\n==== REQUEST (MULTIPART) ====\n" << url << "\nfile: " << file_path << "\ntop_k: " << top_k << "\n";
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
        if (logFile.is_open()) { logFile << "\n==== RESPONSE ====\n" << response << "\n"; }
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
New_model_name=Blat 1
New_dwx_path=dodatki\akcesoria_kuchenne\blat_i
New_jpg_path=dodatki\akcesoria_kuchenne\blat_i.jpg
)";
            create_config.close();
        }
    } else {
        check_config.close();
    }

    int choice = -1;

    while (choice != 0) {
        menu();
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

        auto config = load_config(config_filename);
        std::string base_url = config["base_url"];
        std::string top_k = config["top_k"];
        std::string directories = config["directories"];
        fs::path db_path = "../../../clip_fast_api/data/models.db";
        fs::path absolute_db_path = fs::absolute(db_path);

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
                // 🚀 ROZBUDOWANA DIAGNOSTYKA I INTERFEJS CMD DLA ALFY (Wyszukiwanie Hybrydowe)
                std::string local_input;
                std::cout << "Enter text search query (e.g., modern oak chair): ";
                std::getline(std::cin, local_input);
                
                if (local_input.empty() || std::all_of(local_input.begin(), local_input.end(), [](unsigned char c) { return std::isspace(c); })) {
                    std::cout << "⚠️ Search query cannot be empty!\n";
                    break;
                }

                // Pobieranie wartości alfa live z terminala CMD
                std::cout << "Enter Alpha balance for Score Fusion [0.0 = 100% Text, 1.0 = 100% Image] (default: 0.35): ";
                std::string alpha_input;
                std::getline(std::cin, alpha_input);
                
                double alpha = 0.35; // Sugerowany punkt startowy
                if (!alpha_input.empty()) {
                    try {
                        double parsed = std::stod(alpha_input);
                        if (parsed >= 0.0 && parsed <= 1.0) {
                            alpha = parsed;
                        } else {
                            std::cout << "⚠️ Alpha out of range [0.0 - 1.0]. Using default 0.35\n";
                        }
                    } catch (...) {
                        std::cout << "⚠️ Invalid input type. Using default 0.35\n";
                    }
                }

                std::string search_query = console_to_utf8(local_input);
                std::string escaped_query = safe_json_string(search_query);
                
                std::wcout << L"Executing Hybrid Lookup for: '" << utf8_to_wstring(search_query) << L"' with Alpha = " << alpha << L"...\n";
                
                // Budowa payloadu JSON uwzględniającego parametr alpha
                std::string json_payload = "{\"text\": \"" + escaped_query + "\", \"top_k\": " + top_k + ", \"alpha\": " + std::to_string(alpha) + "}";
                
                // Wysyłamy żądanie do nowego, zunifikowanego endpointu hybrydowego
                std::string response = send_json_post(curl, base_url + "/find-similar-images-by-hybrid-search", json_payload);
                process_and_display_results(response);
                break;
            }

            case 3: {
                std::string image_input = config["target_image"];
                if (image_input.empty()) {
                    std::cout << "Error: 'target_image' is not defined in config.txt!\n";
                    break;
                }
                
                std::cout << "Executing visual lookup for asset from config: '" << image_input << "'...\n";
                std::string response = send_multipart_post(curl, base_url + "/find-similar-images-by-image", image_input, top_k);
                process_and_display_results(response);
                break;
            }

            case 4: {
                std::string name, dwx_path, jpg_path, manufacturer = "Unknown";
                std::cout << "\n--- Add New Model ---\n";

                name = config["New_model_name"];
                dwx_path = config["New_dwx_path"];
                jpg_path = config["New_jpg_path"];

                if (dwx_path.empty() || name.empty()) {
                    std::cout << "⚠️ Error: Name and DWX Path cannot be empty!\n";
                    break;
                }

                sqlite3* db = nullptr;
                if (sqlite3_open(absolute_db_path.string().c_str(), &db) == SQLITE_OK) {
                    AddNewModelToDB(db, name, manufacturer, dwx_path, jpg_path, base_url);
                    sqlite3_close(db);
                    std::cout << "🚀 Add operation complete. Database connection released.\n";
                } else {
                    std::cerr << "❌ Cannot open database: " << sqlite3_errmsg(db) << "\n";
                }
                break;
            }

            case 5: {
                std::string dwx_path_to_delete;
                std::cout << "\n--- Delete Model Manually ---\n";
                std::cout << "Enter the exact (.dwx) path of the model to remove: ";
                std::getline(std::cin, dwx_path_to_delete);

                if (dwx_path_to_delete.empty()) {
                    std::cout << "⚠️ Error: Path cannot be empty!\n";
                    break;
                }

                sqlite3* db = nullptr;
                if (sqlite3_open(absolute_db_path.string().c_str(), &db) == SQLITE_OK) {
                    DeleteModelFromDB(db, dwx_path_to_delete, base_url);
                    sqlite3_close(db);
                    std::cout << "🚀 Delete operation complete. Database connection released.\n";
                } else {
                    std::cerr << "❌ Cannot open database: " << sqlite3_errmsg(db) << "\n";
                }
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