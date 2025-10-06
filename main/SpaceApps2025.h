#pragma once

#include <iostream>
#include <filesystem>
#include "BuildConfig.h"
#include "../data/dataset_split.h"
#include "../web/web_manager.h"
#include "../models/DNN.h"
#include "../data/data_manager.h"

void train_model(std::string model) {
	data::data_manager dm;
	if (model == "K2") {
		try {
			std::filesystem::path sourcePathFs(SOURCE_PATH);
			// Make sure your data file is named correctly
			auto k2Path = sourcePathFs / "data" / "k2.csv";

			std::cout << "Loading, shuffling, and splitting K2 data..." << std::endl;
			auto datasets = dm.load_and_split_k2_data(k2Path.string(), 0.8f);

			std::cout << "Data loaded successfully." << std::endl;
			std::cout << "Training samples: " << datasets.training_inputs.size() << std::endl;
			std::cout << "Testing samples:  " << datasets.testing_inputs.size() << std::endl;

			if (datasets.training_inputs.empty()) {
				std::cerr << "No training data loaded. Exiting." << std::endl;
				return;
			}

			// --- 1. Define Hyperparameters for the Hybrid Model ---

			// Input dimensions are derived from the data
			int input_sequence_length = datasets.training_inputs[0].size();
			int num_classes = datasets.training_targets[0].size();

			// Architectural choices for the new layers
			std::vector<int> hidden_layers = { 64, 32 };
			int num_filters = 16;
			int filter_size = 3;
			int attention_dim = 24;

			// Sanity check: The input sequence must be at least as long as the filter
			if (input_sequence_length < filter_size) {
				throw std::runtime_error("Input sequence length is smaller than the filter size!");
			}

			// --- 2. Use the Correct Constructor for the Hybrid CNN-Attention Model ---
			DNN network(input_sequence_length, num_classes, hidden_layers,
				num_filters, filter_size, attention_dim);

			std::cout << "\nStarting training on K2 data with Hybrid CNN-Attention model..." << std::endl;
			network.train(datasets.training_inputs, datasets.training_targets, 500, 0.005f);
			std::cout << "Training complete." << std::endl;
			std::cout << "--------------------" << std::endl;

			// --- 3. Evaluate the network on the testing data ---
			std::cout << "Evaluating network on testing data..." << std::endl;
			int correct_predictions = 0;
			for (size_t i = 0; i < datasets.testing_inputs.size(); ++i) {
				const auto& input = datasets.testing_inputs[i];
				const auto& target = datasets.testing_targets[i];

				std::vector<float> prediction = network.forward(input);

				// Determine the predicted class by finding the index of the max probability
				int predicted_class = (prediction[0] > prediction[1]) ? 0 : 1;
				int target_class = (target[0] > target[1]) ? 0 : 1;

				if (predicted_class == target_class) {
					correct_predictions++;
				}
			}

			double accuracy = 100.0 * static_cast<double>(correct_predictions) / datasets.testing_inputs.size();
			std::cout << "Final Testing Accuracy: " << accuracy << "%" << std::endl;

		}
		catch (const std::exception& e) {
			std::cerr << "An error occurred: " << e.what() << std::endl;
		}
	}
}