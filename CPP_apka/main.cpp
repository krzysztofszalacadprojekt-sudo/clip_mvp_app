#include <iostream>
#include <string>
#include <curl/curl.h>
#include <fstream>
#include <map>
#include <sstream>

std::ofstream logFile;

void menu() {
    std::cout << "\n==== Menu ====\n";
    std::cout << "1. Update embeddings\n";
    std::cout << "2. Get text embedding\n";
    std::cout << "3. Search by text\n";
    std::cout << "4. Search by image\n";
    std::cout << "0. Exit\n";
    std::cout << "Choose an option: ";
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

void send_json_post(CURL* curl, const std::string& url, const std::string& json_payload) {

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
}

void send_empty_post(CURL* curl, const std::string& url) {
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
}

void send_multipart_post(CURL* curl, const std::string& url, const std::string& file_path, const std::string& top_k) {
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
        return;
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

    auto config = load_config("config.txt");

    std::string base_url = config["base_url"];
    std::string image_path = config["image_path"];
    std::string top_k = config["top_k"];
    std::string directories = config["directories"];
    std::string text = config["text"];

    int choice = -1;

    while (choice != 0) {
        menu();
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            choice = -1;
        }

        switch (choice) {
            case 1: {
                std::cout << "Updating embeddings...\n";
                std::string json_payload = build_directories_json(directories);
                send_json_post(
                    curl,
                    base_url + "/update-embeddings",
                    json_payload
                );
                std::cout << "Done!\n";
                break;
            }

            case 2:
                std::cout << "Getting embedding for text...\n";
                send_json_post(curl, base_url + "/get-embedding", "{\"text\": \"" + text + "\", \"top_k\": " + top_k + "}");
                break;

            case 3:
                std::cout << "Searching by text...\n";
                send_json_post(curl, base_url + "/find-similar-images-by-text", "{\"text\": \"" + text + "\", \"top_k\": " + top_k + "}");
                break;

            case 4:
                std::cout << "Searching by image...\n";
                send_multipart_post(curl, base_url + "/find-similar-images-by-image", image_path, top_k);
                break;

            case 0:
                std::cout << "Exiting...\n";
                break;

            default:
                std::cout << "Invalid choice.\n";
        }
    }

    // if (curl) {
    //     send_json_post(curl, base_url + "/update-embeddings",
    //         "{\"directories\": [\"" + directories + "\"]}");

    //     send_json_post(curl, base_url + "/get-embedding",
    //         "{\"text\": \"" + text + "\", \"top_k\": " + top_k + "}");

    //     send_json_post(curl, base_url + "/find-similar-images-by-text",
    //         "{\"text\": \"" + text + "\", \"top_k\": " + top_k + "}");

    //     send_multipart_post(curl,
    //         base_url + "/find-similar-images-by-image",
    //         image_path,
    //         top_k);

    //     curl_easy_cleanup(curl);
    // }

    if (curl) {
        curl_easy_cleanup(curl);
    }
    logFile.close();
    return 0;
}