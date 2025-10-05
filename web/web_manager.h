#pragma once
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
#include "../data/data_manager.h"

namespace server {
	class web_manager {
	public:
		int run_server();
		const crow::json::rvalue& get_last_data() const;

	private:
		void open_browser(const std::string& url);
		mutable std::mutex data_mutex_;
		crow::json::rvalue last_data_;
	};
} // namespace server