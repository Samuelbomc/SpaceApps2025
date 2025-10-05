#include "data_manager.h"

namespace data {
	// Helper to split a CSV line into a vector of strings
	std::vector<std::string> data_manager::split_csv_line(const std::string& line) {
		std::vector<std::string> result;
		std::stringstream ss(line);
		std::string item;
		while (std::getline(ss, item, ',')) {
			result.push_back(item);
		}
		return result;
	}

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

	DatasetSplit data_manager::load_and_split_k2_data(const std::string& filepath, float train_split_ratio) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file: " + filepath);
		}

		// --- 1. Load ALL raw data and targets ---
		std::vector<std::vector<std::string>> all_raw_features;
		std::vector<std::vector<float>> all_targets;
		std::string line;
		std::getline(file, line); // Skip header

		while (std::getline(file, line)) {
			auto row = split_csv_line(line);
			if (row.size() <= k2_useful_feature_indices.back()) continue;

			std::vector<std::string> feature_row;
			for (int idx : k2_useful_feature_indices) {
				feature_row.push_back(row[idx]);
			}
			all_raw_features.push_back(feature_row);

			if (row[k2_target_column_index] == "CONFIRMED") {
				all_targets.push_back({ 1.0f, 0.0f });
			}
			else {
				all_targets.push_back({ 0.0f, 1.0f });
			}
		}

		if (all_raw_features.empty()) {
			throw std::runtime_error("No valid data found in the file.");
		}

		// --- 2. Shuffle indices ---
		std::vector<int> indices(all_raw_features.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), std::mt19937{ std::random_device{}() });

		// --- 3. Split the raw data ---
		int train_size = static_cast<int>(all_raw_features.size() * train_split_ratio);
		std::vector<std::vector<std::string>> raw_training_features;
		std::vector<std::vector<float>> training_targets;
		std::vector<std::vector<std::string>> raw_testing_features;
		std::vector<std::vector<float>> testing_targets;

		for (size_t i = 0; i < all_raw_features.size(); ++i) {
			int shuffled_idx = indices[i];
			if (i < train_size) {
				raw_training_features.push_back(all_raw_features[shuffled_idx]);
				training_targets.push_back(all_targets[shuffled_idx]);
			}
			else {
				raw_testing_features.push_back(all_raw_features[shuffled_idx]);
				testing_targets.push_back(all_targets[shuffled_idx]);
			}
		}

		// --- 4. Calculate normalization stats ONLY from training data ---
		int num_features = k2_useful_feature_indices.size();
		feature_means.assign(num_features, 0.0f);
		feature_std_devs.assign(num_features, 0.0f);
		std::vector<int> counts(num_features, 0);

		for (const auto& row : raw_training_features) {
			for (int i = 0; i < num_features; ++i) {
				if (!row[i].empty()) {
					try {
						feature_means[i] += std::stof(row[i]);
						counts[i]++;
					}
					catch (...) {}
				}
			}
		}
		for (int i = 0; i < num_features; ++i) {
			if (counts[i] > 0) feature_means[i] /= counts[i];
		}

		counts.assign(num_features, 0);
		for (const auto& row : raw_training_features) {
			for (int i = 0; i < num_features; ++i) {
				if (!row[i].empty()) {
					try {
						float val = std::stof(row[i]);
						feature_std_devs[i] += (val - feature_means[i]) * (val - feature_means[i]);
						counts[i]++;
					}
					catch (...) {}
				}
			}
		}
		for (int i = 0; i < num_features; ++i) {
			if (counts[i] > 1) {
				feature_std_devs[i] = std::sqrt(feature_std_devs[i] / (counts[i] - 1));
			}
			else {
				feature_std_devs[i] = 1.0f;
			}
		}

		// --- 5. Normalize both datasets and return ---
		DatasetSplit result;
		result.training_targets = training_targets;
		result.testing_targets = testing_targets;

		for (const auto& row : raw_training_features) {
			std::vector<float> processed_row;
			for (int i = 0; i < num_features; ++i) {
				float value = feature_means[i];
				if (!row[i].empty()) { try { value = std::stof(row[i]); } catch (...) {} }
				processed_row.push_back((value - feature_means[i]) / (feature_std_devs[i] + 1e-9));
			}
			result.training_inputs.push_back(processed_row);
		}

		for (const auto& row : raw_testing_features) {
			std::vector<float> processed_row;
			for (int i = 0; i < num_features; ++i) {
				float value = feature_means[i];
				if (!row[i].empty()) { try { value = std::stof(row[i]); } catch (...) {} }
				processed_row.push_back((value - feature_means[i]) / (feature_std_devs[i] + 1e-9));
			}
			result.testing_inputs.push_back(processed_row);
		}

		return result;
	}
} // namespace data