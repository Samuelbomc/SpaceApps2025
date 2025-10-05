#pragma once
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include "crow.h"
#include "BuildConfig.h"
#include "../web/web_manager.h"

namespace server {
	class web_manager;
}

namespace data {
	class data_manager {
	public:
		int read_user_input(server::web_manager& mgr);
    private:
        /**
         * @brief Converts a crow::json::rvalue to its string representation.
         */
        std::string json_value_to_string(const crow::json::rvalue& val);

        /**
         * @brief Escapes a string for proper CSV formatting according to RFC 4180.
         *
         * This handles values containing commas, newlines, or double quotes.
         */
        std::string escape_csv(const std::string& value);

	};
}

