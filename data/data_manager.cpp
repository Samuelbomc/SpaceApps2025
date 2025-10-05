#include "data_manager.h"

namespace data {
	int data_manager::read_user_input(server::web_manager& mgr) {
		const auto& data = mgr.get_last_data();
		if (data.t() != crow::json::type::Object) {
			std::cerr << "Error: Input data is not a JSON object." << std::endl;
			return -1;
		}

		std::filesystem::path csv_path = std::filesystem::path(SOURCE_PATH) / "data" / "inputs.csv";

		// --- 1. Define a consistent column order ---
		const std::vector<std::string> headers = {
			"name", "age"
		};

		// --- 2. Check if we need to write the header ---
		bool needs_header = !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;

		std::ofstream csv_file(csv_path, std::ios::app);
		if (!csv_file.is_open()) {
			std::cerr << "Error: Could not open CSV file at " << csv_path << std::endl;
			return -1;
		}

		// --- 3. Write the header row if needed ---
		if (needs_header) {
			for (size_t i = 0; i < headers.size(); ++i) {
				csv_file << headers[i] << (i < headers.size() - 1 ? "," : "");
			}
			csv_file << "\n";
		}

		// --- 4. Write the data row in the correct order ---
		for (size_t i = 0; i < headers.size(); ++i) {
			const std::string& key = headers[i];
			std::string value_str;

			if (data.has(key)) {
				value_str = json_value_to_string(data[key]);
			}

			csv_file << escape_csv(value_str) << (i < headers.size() - 1 ? "," : "");
		}
		csv_file << "\n";

		csv_file.close();
		return 0;
	}

	std::string data_manager::json_value_to_string(const crow::json::rvalue& val) {
		if (val.t() == crow::json::type::String) {
			return val.s();
		}
		// For numbers, bools, etc., use a stringstream for serialization.
		std::stringstream ss;
		ss << val;
		return ss.str();
	}

	std::string data_manager::escape_csv(const std::string& value) {
		// If the value doesn't contain special characters, return it as is.
		if (value.find_first_of(",\"\n") == std::string::npos) {
			return value;
		}

		std::string escaped = "\""; // Enclose in double quotes
		for (char c : value) {
			if (c == '"') {
				escaped += "\"\""; // Escape internal quotes by doubling them
			}
			else {
				escaped += c;
			}
		}
		escaped += "\"";

		return escaped;
	}
} // namespace data