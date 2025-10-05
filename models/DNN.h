#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

class DNN {
public:
    DNN(int input_sequence_length, int num_classes,
        const std::vector<int>& hidden_layers,
        int num_filters, int filter_size, int attention_dim);

    std::vector<float> forward(const std::vector<float>& input);

    void train(const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& targets,
        int epochs, float learning_rate);

private:
    // --- Architecture Parameters ---
    int num_filters;
    int filter_size;
    int attention_dim;

    // --- Learnable Parameters ---
    // Convolutional Layer
    std::vector<float> conv_weights;
    std::vector<float> conv_biases;
    // Attention Mechanism (on the first dense layer)
    std::vector<float> W_q;
    std::vector<float> W_k;
    // Dense (Fully Connected) Layers
    std::vector<std::vector<float>> dense_weights;
    std::vector<std::vector<float>> dense_biases;

    // --- Helper Functions ---
    std::mt19937 generator;
    std::vector<float> xavier_weights(int input_size, int output_size);
    std::vector<float> zero_biases(int size);
    std::vector<float> tanh_derivative(const std::vector<float>& activated_output);
    float sigmoid_derivative(float activated_output);
};
