#include "SpaceApps2025.h"

void print_vector(const std::vector<float>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

int main() {
    // XOR problem
    std::vector<std::vector<float>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    std::vector<std::vector<float>> targets = {
        {1, 0}, // 0
        {0, 1}, // 1
        {0, 1}, // 1
        {1, 0}  // 0
    };

    int input_size = 2;
    int output_size = 2;
    std::vector<int> hidden_layers = { 3 };

    DNN nn(input_size, output_size, hidden_layers);

    int epochs = 400;
    float learning_rate = 0.1;

    std::cout << "Starting training..." << std::endl;
    nn.train(inputs, targets, epochs, learning_rate);
    std::cout << "Training complete." << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << "Testing predictions:" << std::endl;

    for (const auto& input : inputs) {
        std::vector<float> prediction = nn.forward(input);
        std::cout << "Input: ";
        print_vector(input);
        std::cout << "Prediction: ";
        print_vector(prediction);
    }

    return 0;
}
