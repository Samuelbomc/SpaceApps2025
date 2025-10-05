#include "data_manager.h"

namespace data {
	int data_manager::read_user_input(server::web_manager& mgr) {
		auto& data = mgr.get_last_data();

		std::filesystem::path csv_path = std::filesystem::path(SOURCE_PATH) / "data" / "inputs.csv";

		std::ofstream csv_file(csv_path, std::ios::app);
		if (!csv_file.is_open()) {
			return -1;
		}

		if (data.t() == crow::json::type::Object) {
			bool first = true;
			for (const auto& kv : data) {
				if (!first) csv_file << ",";
				csv_file << '"' << kv.key() << "\":";

				if (kv.t() == crow::json::type::String) {
					csv_file << '"' << kv.s() << '"';
				} else {
					// Serialize non-string types using a stringstream
					std::stringstream ss;
					ss << kv;
					csv_file << '"' << ss.str() << '"';
				}
				first = false;
			}
			csv_file << "\n";
		}

		csv_file.close();
		return 0;
	}
} // namespace data