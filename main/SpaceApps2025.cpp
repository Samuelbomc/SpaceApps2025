#include "crow.h"
#include "crow/middlewares/cors.h"
#include "BuildConfig.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>

void openBrowser(const std::string& url) {
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

int main()
{
    // Initialize a Crow application instance
    crow::App<crow::CORSHandler> app;

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

    // A simple GET route with a dynamic parameter
    CROW_ROUTE(app, "/hello/<string>")([](const std::string& name) {
        std::string response_message = "Hello, " + name + "!";
        return crow::response(200, response_message);
        });

    // --- POST Route ---
    CROW_ROUTE(app, "/api/process").methods("POST"_method)
        ([](const crow::request& req) {
        // Load and validate the JSON from the request body
        auto data = crow::json::load(req.body);
        if (!data || !data.has("name")) {
            return crow::response(400, "Invalid JSON or missing 'name' field.");
        }

        // Safely extract data from the JSON object
        std::string user_name = data["name"].s();
        int user_age = data.has("age") ? data["age"].i() : 0;

        std::cout << "Received data: Name = " << user_name << ", Age = " << user_age << std::endl;

        // Prepare a JSON response
        crow::json::wvalue response_json;
        response_json["status"] = "success";
        response_json["message"] = "Processed data for " + user_name + ".";
        response_json["timestamp"] = time(nullptr);

        // Crow automatically sets the Content-Type to application/json
        return crow::response(200, response_json);
            });

    // --- Server Startup Logic ---
    const int PORT = 18080;
    const std::string SERVER_URL = "http://localhost:" + std::to_string(PORT) + "/";

    // Run the server in a separate thread.
    std::thread server_thread([&]() {
        std::cout << "Server starting on port " << PORT << "..." << std::endl;
        app.port(PORT).multithreaded().run();
        });

    // Open the browser in a separate thread.
    std::thread browser_thread(openBrowser, SERVER_URL);

    // Wait for both threads to complete.
    server_thread.join();
    browser_thread.join();

    return 0;
}

