#include "DNN.h"

// --- Constructor ---
DNN::DNN(int input_size, int output_size, const std::vector<int>& hidden_layers) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->hidden_layers = hidden_layers;
    this->generator = std::mt19937(std::random_device{}());

    int last_layer_size = input_size;

    // Initialize hidden layers
    if (!hidden_layers.empty()) {
        for (size_t i = 0; i < hidden_layers.size(); ++i) {
            int hidden_size = hidden_layers[i];
            weights.push_back(xavier_weights(last_layer_size, hidden_size));
            biases.push_back(zero_biases(hidden_size));
            last_layer_size = hidden_size;

            // Initialize attention weights after the first hidden layer
            if (i == 0) {
                this->attention_dim = hidden_size;
                W_q = xavier_weights(hidden_size, attention_dim);
                W_k = xavier_weights(hidden_size, attention_dim);
                W_v = xavier_weights(hidden_size, attention_dim);
            }
        }
    }

    // Initialize output layer
    weights.push_back(xavier_weights(last_layer_size, output_size));
    biases.push_back(zero_biases(output_size));
}

// --- Forward Pass ---
std::vector<float> DNN::forward(const std::vector<float>& input) {
    std::vector<float> current_activation = input;

    for (size_t layer = 0; layer < weights.size(); ++layer) {
        const auto& w = weights[layer];
        const auto& b = biases[layer];
        int layer_output_size = b.size();
        int layer_input_size = current_activation.size();
        std::vector<float> z(layer_output_size, 0.0f);

        // Compute z = W * a + b
        for (int j = 0; j < layer_output_size; ++j) {
            for (int k = 0; k < layer_input_size; ++k) {
                z[j] += w[j * layer_input_size + k] * current_activation[k];
            }
            z[j] += b[j];
        }

        // Apply activation function
        if (layer == weights.size() - 1) { // Output layer
            float max_z = *std::max_element(z.begin(), z.end());
            float sum_exp = 0.0f;
            for (float val : z) sum_exp += std::exp(val - max_z);
            for (float& val : z) val = std::exp(val - max_z) / sum_exp;
        }
        else { // Hidden layers
            for (float& val : z) val = std::tanh(val);
        }
        current_activation = z;

        // Apply self-attention after the first hidden layer
        if (layer == 0 && !hidden_layers.empty()) {
            const auto& hidden_activations = current_activation;
            int hidden_size = hidden_activations.size();

            std::vector<float> query(attention_dim, 0.0f), key(attention_dim, 0.0f), value(attention_dim, 0.0f);
            for (int j = 0; j < attention_dim; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    query[j] += W_q[j * hidden_size + k] * hidden_activations[k];
                    key[j] += W_k[j * hidden_size + k] * hidden_activations[k];
                    value[j] += W_v[j * hidden_size + k] * hidden_activations[k];
                }
            }

            float score = 0.0f;
            for (int j = 0; j < attention_dim; ++j) score += query[j] * key[j];
            score /= std::sqrt(static_cast<float>(attention_dim));
            float attention_weight = 1.0f / (1.0f + std::exp(-score));

            std::vector<float> context(attention_dim, 0.0f);
            for (int j = 0; j < attention_dim; ++j) context[j] = value[j] * attention_weight;

            current_activation = context;
        }
    }
    return current_activation;
}

// --- Training Loop ---
void DNN::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        std::vector<int> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int i : indices) {
            const auto& input = inputs[i];
            const auto& target = targets[i];

            // --- 1. Forward Pass (storing intermediate values) ---
            std::vector<std::vector<float>> activations;
            activations.push_back(input);

            std::vector<float> hidden_activations_pre_attention;
            std::vector<float> query, key, value;
            float score = 0.0f, attention_weight = 0.0f;

            std::vector<float> current_activation = input;

            for (size_t layer = 0; layer < weights.size(); ++layer) {
                const auto& w = weights[layer];
                const auto& b = biases[layer];
                int layer_output_size = b.size();
                int layer_input_size = current_activation.size();
                std::vector<float> z(layer_output_size, 0.0f);
                for (int j = 0; j < layer_output_size; ++j) {
                    for (int k = 0; k < layer_input_size; ++k) {
                        z[j] += w[j * layer_input_size + k] * current_activation[k];
                    }
                    z[j] += b[j];
                }

                if (layer == weights.size() - 1) {
                    float max_z = *std::max_element(z.begin(), z.end());
                    float sum_exp = 0.0f;
                    for (float val : z) sum_exp += std::exp(val - max_z);
                    for (float& val : z) val = std::exp(val - max_z) / sum_exp;
                }
                else {
                    for (float& val : z) val = std::tanh(val);
                }
                current_activation = z;

                if (layer == 0 && !hidden_layers.empty()) {
                    hidden_activations_pre_attention = current_activation;
                    int hidden_size = hidden_activations_pre_attention.size();

                    query.assign(attention_dim, 0.0f);
                    key.assign(attention_dim, 0.0f);
                    value.assign(attention_dim, 0.0f);
                    for (int j = 0; j < attention_dim; ++j) {
                        for (int k = 0; k < hidden_size; ++k) {
                            query[j] += W_q[j * hidden_size + k] * hidden_activations_pre_attention[k];
                            key[j] += W_k[j * hidden_size + k] * hidden_activations_pre_attention[k];
                            value[j] += W_v[j * hidden_size + k] * hidden_activations_pre_attention[k];
                        }
                    }

                    score = 0.0f;
                    for (int j = 0; j < attention_dim; ++j) score += query[j] * key[j];
                    score /= std::sqrt(static_cast<float>(attention_dim));
                    attention_weight = 1.0f / (1.0f + std::exp(-score));

                    std::vector<float> context(attention_dim, 0.0f);
                    for (int j = 0; j < attention_dim; ++j) context[j] = value[j] * attention_weight;

                    current_activation = context;
                }
                activations.push_back(current_activation);
            }

            for (size_t j = 0; j < target.size(); ++j) {
                total_loss -= target[j] * std::log(activations.back()[j] + 1e-9);
            }

            // --- 2. Backward Pass (Corrected Logic) ---
            std::vector<float> delta = activations.back();
            for (size_t j = 0; j < delta.size(); ++j) delta[j] -= target[j];

            for (int layer = weights.size() - 1; layer >= 0; --layer) {
                const auto& current_activations = activations[layer + 1];
                const auto& prev_activations = activations[layer];

                // Calculate gradients for current layer's weights and biases
                std::vector<float> nabla_b = delta;
                std::vector<float> nabla_w(prev_activations.size() * delta.size());
                for (size_t j = 0; j < delta.size(); ++j) {
                    for (size_t k = 0; k < prev_activations.size(); ++k) {
                        nabla_w[j * prev_activations.size() + k] = delta[j] * prev_activations[k];
                    }
                }

                // Calculate delta for the previous layer's output (before activation derivative)
                std::vector<float> prev_delta(prev_activations.size(), 0.0f);
                if (layer > 0) {
                    for (size_t j = 0; j < prev_activations.size(); ++j) {
                        for (size_t k = 0; k < delta.size(); ++k) {
                            prev_delta[j] += weights[layer][k * prev_activations.size() + j] * delta[k];
                        }
                    }
                }

                // Update current layer's weights and biases
                for (size_t j = 0; j < weights[layer].size(); ++j) weights[layer][j] -= learning_rate * nabla_w[j];
                for (size_t j = 0; j < biases[layer].size(); ++j) biases[layer][j] -= learning_rate * nabla_b[j];

                // If the previous layer's output was from the attention mechanism,
                // backpropagate through attention.
                if (layer == 1 && !hidden_layers.empty()) {
                    auto& delta_for_context = prev_delta;
                    int hidden_size = hidden_activations_pre_attention.size();

                    float nabla_attention_weight = 0.0f;
                    for (int j = 0; j < attention_dim; ++j) nabla_attention_weight += delta_for_context[j] * value[j];

                    std::vector<float> nabla_value(attention_dim);
                    for (int j = 0; j < attention_dim; ++j) nabla_value[j] = delta_for_context[j] * attention_weight;

                    float nabla_score = nabla_attention_weight * sigmoid_derivative(attention_weight);

                    float scale = 1.0f / std::sqrt(static_cast<float>(attention_dim));
                    std::vector<float> nabla_query(attention_dim), nabla_key(attention_dim);
                    for (int j = 0; j < attention_dim; ++j) {
                        nabla_query[j] = nabla_score * key[j] * scale;
                        nabla_key[j] = nabla_score * query[j] * scale;
                    }

                    std::vector<float> nabla_Wq(W_q.size()), nabla_Wk(W_k.size()), nabla_Wv(W_v.size());
                    for (int j = 0; j < attention_dim; ++j) {
                        for (int k = 0; k < hidden_size; ++k) {
                            nabla_Wq[j * hidden_size + k] = nabla_query[j] * hidden_activations_pre_attention[k];
                            nabla_Wk[j * hidden_size + k] = nabla_key[j] * hidden_activations_pre_attention[k];
                            nabla_Wv[j * hidden_size + k] = nabla_value[j] * hidden_activations_pre_attention[k];
                        }
                    }

                    std::vector<float> delta_pre_attention(hidden_size, 0.0f);
                    for (int j = 0; j < hidden_size; ++j) { for (int k = 0; k < attention_dim; ++k) { delta_pre_attention[j] += W_q[k * hidden_size + j] * nabla_query[k]; } }
                    for (int j = 0; j < hidden_size; ++j) { for (int k = 0; k < attention_dim; ++k) { delta_pre_attention[j] += W_k[k * hidden_size + j] * nabla_key[k]; } }
                    for (int j = 0; j < hidden_size; ++j) { for (int k = 0; k < attention_dim; ++k) { delta_pre_attention[j] += W_v[k * hidden_size + j] * nabla_value[k]; } }

                    for (size_t j = 0; j < W_q.size(); ++j) W_q[j] -= learning_rate * nabla_Wq[j];
                    for (size_t j = 0; j < W_k.size(); ++j) W_k[j] -= learning_rate * nabla_Wk[j];
                    for (size_t j = 0; j < W_v.size(); ++j) W_v[j] -= learning_rate * nabla_Wv[j];

                    prev_delta = delta_pre_attention;
                }

                // Apply the activation derivative to the delta
                if (layer > 0) {
                    const auto& derivative = tanh_derivative((layer == 1 && !hidden_layers.empty()) ? hidden_activations_pre_attention : current_activations);
                    for (size_t j = 0; j < prev_delta.size(); ++j) {
                        prev_delta[j] *= derivative[j];
                    }
                }
                delta = prev_delta;
            }
        }
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

// --- Helper Implementations ---
std::vector<float> DNN::xavier_weights(int input_size, int output_size) {
    float limit = std::sqrt(6.0f / (input_size + output_size));
    std::uniform_real_distribution<float> dist(-limit, limit);
    std::vector<float> w(input_size * output_size);
    for (auto& weight : w) {
        weight = dist(this->generator);
    }
    return w;
}

std::vector<float> DNN::zero_biases(int size) {
    return std::vector<float>(size, 0.0f);
}

std::vector<float> DNN::tanh_derivative(const std::vector<float>& activated_output) {
    std::vector<float> derivative(activated_output.size());
    for (size_t i = 0; i < activated_output.size(); ++i) {
        derivative[i] = 1.0f - activated_output[i] * activated_output[i];
    }
    return derivative;
}

float DNN::sigmoid_derivative(float activated_output) {
    return activated_output * (1.0f - activated_output);
}

