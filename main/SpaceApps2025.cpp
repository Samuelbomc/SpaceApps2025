#include "SpaceApps2025.h"

int main() {
    // --- Create a DNN for the XOR problem ---
    // Input size: 2, Output size: 2 (for [1,0] and [0,1]), Hidden Layers: 1 layer of 3 neurons
    DNN network(2, 2, { 3 });

    // --- Training Data (XOR) ---
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };
    // One-hot encoded targets: 0 -> [1,0], 1 -> [0,1]
    std::vector<std::vector<float>> targets = {
        {1.0f, 0.0f}, // 0
        {0.0f, 1.0f}, // 1
        {1.0f, 0.0f}  // 0
    };

    // --- Train the network ---
    std::cout << "Starting training..." << std::endl;
    network.train(inputs, targets, 200, 0.1f);
    std::cout << "Training complete." << std::endl;
    std::cout << "--------------------" << std::endl;

    // --- Test the trained network ---
    std::cout << "Testing predictions:" << std::endl;
    for (const auto& input : inputs) {
        std::vector<float> prediction = network.forward(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "], Prediction: [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;
    }

    return 0;
}