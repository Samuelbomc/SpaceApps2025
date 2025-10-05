#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

class DNN {
public:
	/*
	 * Constructor for the DNN class.
	 * @param input_size Size of the input layer.
	 * @param output_size Size of the output layer.
	 * @param hidden_layers Vector containing the sizes of the hidden layers.
	 * Activation function: Tanh for hidden layers, Softmax for output layer.
	 */
	DNN(int input_size, int output_size, const std::vector<int>& hidden_layers);
	std::vector<float> forward(const std::vector<float>& input);
	void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate);
private:
	std::vector<float> zero_biases(int size);
	std::vector<float> xavier_weights(int input_size, int output_size);
	std::vector<float> tanh_derivative(const std::vector<float>& activated_output);
	float sigmoid_derivative(float activated_output);

	std::vector<float> W_q; // Query weights
	std::vector<float> W_k; // Key weights
	std::vector<float> W_v; // Value weights
	int attention_dim;
	
	std::mt19937 generator;
	std::uniform_real_distribution<float> distribution;
	int input_size;
	int output_size;
	std::vector<int> hidden_layers;
	std::vector<std::vector<float>> weights;
	std::vector<std::vector<float>> biases;
};