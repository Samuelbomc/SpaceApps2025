#include "web_manager.h"

namespace server {
    void web_manager::open_browser(const std::string& url) {
        // Wait a moment for the server to start
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "Attempting to open browser at: " << url << std::endl;

    #ifdef _WIN32
        std::string command = "start \"\" \"" + url + "\"";
    #elif __APPLE__
        std::string command = "open " + url;
    #elif __linux__
        std::string command = "xdg-open " + url;
    #else
        std::cout << "Unsupported platform: Cannot open browser automatically." << std::endl;
        return;
    #endif

        system(command.c_str());
    }

    int web_manager::run_server() {
        crow::App<crow::CORSHandler> app;
        data::data_manager data_mgr;

        // --- Configure CORS Middleware ---
        auto& cors = app.get_middleware<crow::CORSHandler>();
        cors
            .global()
            .origin("*")
            .headers("Content-Type")
            .methods("POST"_method, "GET"_method);

        // --- API Routes ---
        const std::filesystem::path htmlFilePath = std::filesystem::path(SOURCE_PATH) / "html" / "index.html";
        const std::string HTML_FILE = htmlFilePath.string();

        CROW_ROUTE(app, "/")([HTML_FILE](const crow::request&, crow::response& res) {
            std::ifstream file(HTML_FILE);
            if (file) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                res.set_header("Content-Type", "text/html");
                res.write(buffer.str());
            }
            else {
                res.code = 404;
                res.write("404 Not Found: Could not open index.html");
            }
            res.end();
            });

        // --- POST Route ---
        CROW_ROUTE(app, "/api/process").methods("POST"_method)
            ([this, &data_mgr](const crow::request& req) {
            auto data = crow::json::load(req.body);
            if (!data) {
                return crow::response(400, "Invalid JSON.");
            }

            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                last_data_ = data;
            }

            data_mgr.read_user_input(*this);

            crow::json::wvalue response_json;
            response_json["status"] = "success";
            response_json["message"] = "Processed data.";
            response_json["timestamp"] = time(nullptr);

            return crow::response(200, response_json);
                });

        // --- Server Startup Logic ---
        const int PORT = 18080;
        const std::string SERVER_URL = "http://localhost:" + std::to_string(PORT) + "/";

        std::thread server_thread([&]() {
            std::cout << "Server starting on port " << PORT << "..." << std::endl;
            app.port(PORT).multithreaded().run();
            });

        std::thread browser_thread(&web_manager::open_browser, this, SERVER_URL);

        server_thread.join();
        browser_thread.join();

        return 0;
    }

    const crow::json::rvalue& web_manager::get_last_data() const {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return last_data_;
    }
} // namespace server