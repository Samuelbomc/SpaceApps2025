#pragma once
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include "dataset_split.h"
#include "crow.h"
#include "BuildConfig.h"
#include "../web/web_manager.h"

// Forward declaration of the struct that will be defined in SpaceApps2025.h
struct DatasetSplit;

namespace server {
	class web_manager;
}

namespace data {
	class data_manager {
	public:
		int read_user_input(server::web_manager& mgr);

		DatasetSplit load_and_split_k2_data(const std::string& filepath, float train_split_ratio = 0.8f);

		DatasetSplit load_and_split_tess_data(const std::string& filepath, float train_split_ratio = 0.8f);

		DatasetSplit load_and_split_kepler_data(const std::string& filepath, float train_split_ratio = 0.8f);

	private:
		std::string json_value_to_string(const crow::json::rvalue& val);
		std::vector<std::string> split_csv_line(const std::string& line);
		std::string escape_csv(const std::string& value);

		// --- Data Preprocessing Parameters ---
		std::vector<float> feature_means;
		std::vector<float> feature_std_devs;

		// --- Column Indices for K2 Dataset ---
		const std::vector<int> k2_useful_feature_indices = {
			13, // pl_orbper
			21, // pl_rade
			29, // pl_bmasse
			41, // pl_insol
			45, // pl_eqt
			53, // st_teff
			57, // st_rad
			61, // st_mass
			69  // st_logg
		};

		const int k2_target_column_index = 3; // disposition column

		// --- Column Indices for TESS Dataset ---
		const std::vector<int> tess_useful_feature_indices = {
			19, // pl_orbper
			23, // pl_trandurh
			31, // pl_rade
			35, // pl_insol
			39, // pl_eqt
			51, // st_teff
			55, // st_logg
			59  // st_rad
		};
		const int tess_target_column_index = 2; // disposition column

		// --- Column Indices for Kepler Dataset ---
		const std::vector<int> kepler_useful_feature_indices = {
			10, // koi_period
			19, // koi_duration
			25, // koi_prad
			28, // koi_teq
			31, // koi_insol
			37, // koi_steff
			40, // koi_slogg
			43  // koi_srad
		};
		const int kepler_target_column_index = 3; // koi_disposition
	};
}

