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

namespace server {
	void open_browser(const std::string& url);
	int run_server();
} // namespace server