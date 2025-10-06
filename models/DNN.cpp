#include "DNN.h"

DNN::DNN(int input_sequence_length, int num_classes,
    const std::vector<int>& hidden_layers,
    int num_filters, int filter_size, int attention_dim) {

    this->num_filters = num_filters;
    this->filter_size = filter_size;
    this->attention_dim = attention_dim;
    this->num_classes = num_classes;
    this->generator = std::mt19937(std::random_device{}());

    // 1. Initialize CNN Layer
    conv_weights = xavier_weights(filter_size, num_filters);
    conv_biases = zero_biases(num_filters);

    // 2. Calculate the size of the flattened layer (input to dense layers)
    int conv_output_length = input_sequence_length - filter_size + 1;
    int pool_output_length = conv_output_length / 2;
    int flattened_size = pool_output_length * num_filters;

    // 3. Initialize Dense and Attention Layers
    if (hidden_layers.empty()) {
        throw std::invalid_argument("Hybrid model requires at least one hidden layer for attention.");
    }

    int last_layer_size = flattened_size;
    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        int hidden_size = hidden_layers[i];
        dense_weights.push_back(xavier_weights(last_layer_size, hidden_size));
        dense_biases.push_back(zero_biases(hidden_size));

        // Initialize attention weights for the first hidden layer
        if (i == 0) {
            W_q = xavier_weights(hidden_size, attention_dim);
            W_k = xavier_weights(hidden_size, attention_dim);
        }
        last_layer_size = hidden_size;
    }
    // Final classification layer
    dense_weights.push_back(xavier_weights(last_layer_size, num_classes));
    dense_biases.push_back(zero_biases(num_classes));
}

std::vector<float> DNN::forward(const std::vector<float>& input) {
    // 1. Convolutional Layer
    int conv_output_length = input.size() - filter_size + 1;
    std::vector<float> conv_output(conv_output_length * num_filters);
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < conv_output_length; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < filter_size; ++j) {
                sum += input[i + j] * conv_weights[f * filter_size + j];
            }
            sum += conv_biases[f];
            conv_output[i * num_filters + f] = std::tanh(sum);
        }
    }

    // 2. Max Pooling Layer
    int pool_output_length = conv_output_length / 2;
    std::vector<float> pool_output(pool_output_length * num_filters);
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < pool_output_length; ++i) {
            // Take the maximum value from a 2-element window for each filter
            pool_output[i * num_filters + f] = std::max(
                conv_output[(i * 2) * num_filters + f],
                conv_output[(i * 2 + 1) * num_filters + f]
            );
        }
    }

    // 3. Dense & Attention Layers
    std::vector<float> current_activation = pool_output;
    for (size_t layer = 0; layer < dense_weights.size(); ++layer) {
        const auto& w = dense_weights[layer];
        const auto& b = dense_biases[layer];
        int output_size = b.size();
        int input_size = current_activation.size();
        std::vector<float> z(output_size, 0.0f);

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                z[i] += w[i * input_size + j] * current_activation[j];
            }
            z[i] += b[i];
        }

        if (layer == dense_weights.size() - 1) { // Softmax for the final classification
            float max_z = *std::max_element(z.begin(), z.end());
            float sum_exp = 0.0f;
            for (float val : z) sum_exp += std::exp(val - max_z);
            for (float& val : z) val = std::exp(val - max_z) / sum_exp;
            current_activation = z;
        }
        else { // Tanh for hidden layers
            for (float& val : z) val = std::tanh(val);
            current_activation = z;
        }

        // Apply self-attention after the first dense layer
        if (layer == 0) {
            const auto& hidden_activations = current_activation;
            int hidden_size = hidden_activations.size();

            std::vector<float> query(attention_dim, 0.0f), key(attention_dim, 0.0f);
            for (int j = 0; j < attention_dim; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    query[j] += W_q[j * hidden_size + k] * hidden_activations[k];
                    key[j] += W_k[j * hidden_size + k] * hidden_activations[k];
                }
            }
            float score = 0.0f;
            for (int j = 0; j < attention_dim; ++j) score += query[j] * key[j];
            score /= std::sqrt(static_cast<float>(attention_dim));

            float attention_weight = 1.0f / (1.0f + std::exp(-score));

            // Apply the gate to the activations
            for (float& val : current_activation) {
                val *= attention_weight;
            }
        }
    }
    return current_activation;
}


void DNN::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        std::vector<int> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int i : indices) {
            const auto& input = inputs[i];
            const auto& target = targets[i];

            // --- 1. FORWARD PASS (storing intermediate values) ---

            // a. Convolutional Layer
            int conv_output_length = input.size() - filter_size + 1;
            std::vector<float> conv_output(conv_output_length * num_filters);
            for (int f = 0; f < num_filters; ++f) {
                for (int j = 0; j < conv_output_length; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < filter_size; ++k) {
                        sum += input[j + k] * conv_weights[f * filter_size + k];
                    }
                    sum += conv_biases[f];
                    conv_output[j * num_filters + f] = std::tanh(sum);
                }
            }

            // b. Max Pooling Layer (storing indices)
            int pool_output_length = conv_output_length / 2;
            std::vector<float> pool_output(pool_output_length * num_filters);
            std::vector<int> max_indices(pool_output_length * num_filters);
            for (int f = 0; f < num_filters; ++f) {
                for (int j = 0; j < pool_output_length; ++j) {
                    int idx1 = (j * 2) * num_filters + f;
                    int idx2 = (j * 2 + 1) * num_filters + f;
                    if (conv_output[idx1] > conv_output[idx2]) {
                        pool_output[j * num_filters + f] = conv_output[idx1];
                        max_indices[j * num_filters + f] = idx1;
                    }
                    else {
                        pool_output[j * num_filters + f] = conv_output[idx2];
                        max_indices[j * num_filters + f] = idx2;
                    }
                }
            }

            // c. Dense & Attention Layers
            std::vector<std::vector<float>> dense_activations;
            dense_activations.push_back(pool_output);
            std::vector<float> current_activation = pool_output;

            std::vector<float> h1_pre_attention;
            float attention_weight = 0.0f;
            std::vector<float> query, key;

            for (size_t layer = 0; layer < dense_weights.size(); ++layer) {
                const auto& w = dense_weights[layer];
                const auto& b = dense_biases[layer];
                int output_size = b.size();
                int input_size = current_activation.size();
                std::vector<float> z(output_size, 0.0f);
                for (int j = 0; j < output_size; ++j) {
                    for (int k = 0; k < input_size; ++k) {
                        z[j] += w[j * input_size + k] * current_activation[k];
                    }
                    z[j] += b[j];
                }

                if (layer == dense_weights.size() - 1) {
                    float max_z = *std::max_element(z.begin(), z.end());
                    float sum_exp = 0.0f;
                    for (float val : z) sum_exp += std::exp(val - max_z);
                    for (float& val : z) val = std::exp(val - max_z) / sum_exp;
                    current_activation = z;
                }
                else {
                    for (float& val : z) val = std::tanh(val);
                    current_activation = z;
                }

                if (layer == 0) {
                    h1_pre_attention = current_activation;
                    int hidden_size = h1_pre_attention.size();
                    query.assign(attention_dim, 0.0f);
                    key.assign(attention_dim, 0.0f);
                    for (int j = 0; j < attention_dim; ++j) {
                        for (int k = 0; k < hidden_size; ++k) {
                            query[j] += W_q[j * hidden_size + k] * h1_pre_attention[k];
                            key[j] += W_k[j * hidden_size + k] * h1_pre_attention[k];
                        }
                    }
                    float score = 0.0f;
                    for (int j = 0; j < attention_dim; ++j) score += query[j] * key[j];
                    score /= std::sqrt(static_cast<float>(attention_dim));
                    attention_weight = 1.0f / (1.0f + std::exp(-score));
                    for (float& val : current_activation) {
                        val *= attention_weight;
                    }
                }
                dense_activations.push_back(current_activation);
            }

            // --- Loss Calculation ---
            const auto& final_prediction = dense_activations.back();
            for (size_t j = 0; j < target.size(); ++j) {
                total_loss -= target[j] * std::log(final_prediction[j] + 1e-9);
            }

            // --- 2. BACKWARD PASS ---

            // a. Initial Delta (from Cross-Entropy Loss + Softmax)
            std::vector<float> delta = final_prediction;
            // ***** CRITICAL BUG FIX *****
            // The original code was hardcoded for 2 classes. This is the general form.
            for (int j = 0; j < this->num_classes; ++j) {
                delta[j] -= target[j];
            }

            std::vector<std::vector<float>> nabla_dense_w(dense_weights.size());
            std::vector<std::vector<float>> nabla_dense_b(dense_biases.size());

            // b. Backprop through Dense Layers (after attention)
            for (int layer = dense_weights.size() - 1; layer > 0; --layer) {
                const auto& prev_activations = dense_activations[layer];
                nabla_dense_b[layer] = delta;
                nabla_dense_w[layer].assign(dense_weights[layer].size(), 0.0f);
                for (size_t j = 0; j < delta.size(); ++j) {
                    for (size_t k = 0; k < prev_activations.size(); ++k) {
                        nabla_dense_w[layer][j * prev_activations.size() + k] = delta[j] * prev_activations[k];
                    }
                }

                const auto& w = dense_weights[layer];
                std::vector<float> prev_delta(prev_activations.size(), 0.0f);
                for (size_t j = 0; j < prev_activations.size(); ++j) {
                    for (size_t k = 0; k < delta.size(); ++k) {
                        prev_delta[j] += w[k * prev_activations.size() + j] * delta[k];
                    }
                }

                const auto& derivative = tanh_derivative(prev_activations);
                for (size_t j = 0; j < prev_delta.size(); ++j) {
                    prev_delta[j] *= derivative[j];
                }
                delta = prev_delta;
            }

            // c. Backprop through Attention Mechanism
            std::vector<float> nabla_h1_post_attention = delta;
            float nabla_attention_weight = 0.0f;
            for (size_t j = 0; j < h1_pre_attention.size(); ++j) {
                nabla_attention_weight += nabla_h1_post_attention[j] * h1_pre_attention[j];
            }
            std::vector<float> delta_pre_attention(h1_pre_attention.size());
            for (size_t j = 0; j < h1_pre_attention.size(); ++j) {
                delta_pre_attention[j] = nabla_h1_post_attention[j] * attention_weight;
            }
            float nabla_score = nabla_attention_weight * sigmoid_derivative(attention_weight);
            float scale = 1.0f / std::sqrt(static_cast<float>(attention_dim));
            std::vector<float> nabla_query(attention_dim), nabla_key(attention_dim);
            for (int j = 0; j < attention_dim; ++j) {
                nabla_query[j] = nabla_score * key[j] * scale;
                nabla_key[j] = nabla_score * query[j] * scale;
            }
            std::vector<float> nabla_Wq(W_q.size(), 0.0f), nabla_Wk(W_k.size(), 0.0f);
            int hidden_size = h1_pre_attention.size();
            for (int j = 0; j < attention_dim; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    nabla_Wq[j * hidden_size + k] = nabla_query[j] * h1_pre_attention[k];
                    nabla_Wk[j * hidden_size + k] = nabla_key[j] * h1_pre_attention[k];
                }
            }
            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < attention_dim; ++k) {
                    delta_pre_attention[j] += W_q[k * hidden_size + j] * nabla_query[k];
                    delta_pre_attention[j] += W_k[k * hidden_size + j] * nabla_key[k];
                }
            }

            // d. Backprop through First Dense Layer (pre-attention)
            const auto& derivative = tanh_derivative(h1_pre_attention);
            for (size_t j = 0; j < delta_pre_attention.size(); ++j) {
                delta_pre_attention[j] *= derivative[j];
            }
            nabla_dense_b[0] = delta_pre_attention;
            nabla_dense_w[0].assign(dense_weights[0].size(), 0.0f);
            for (size_t j = 0; j < nabla_dense_b[0].size(); ++j) {
                for (size_t k = 0; k < pool_output.size(); ++k) {
                    nabla_dense_w[0][j * pool_output.size() + k] = nabla_dense_b[0][j] * pool_output[k];
                }
            }
            const auto& w0 = dense_weights[0];
            std::vector<float> delta_for_pool(pool_output.size(), 0.0f);
            for (size_t j = 0; j < pool_output.size(); ++j) {
                for (size_t k = 0; k < delta_pre_attention.size(); ++k) {
                    delta_for_pool[j] += w0[k * pool_output.size() + j] * delta_pre_attention[k];
                }
            }
            delta = delta_for_pool;

            // e. Backprop through Max Pooling ("Unpooling")
            std::vector<float> delta_post_conv(conv_output.size(), 0.0f);
            for (size_t j = 0; j < max_indices.size(); ++j) {
                delta_post_conv[max_indices[j]] = delta[j];
            }

            // f. Backprop through Conv Activation
            const auto& conv_derivative = tanh_derivative(conv_output);
            for (size_t j = 0; j < delta_post_conv.size(); ++j) {
                delta_post_conv[j] *= conv_derivative[j];
            }

            // g. Backprop through Convolutional Weights (Calculate Gradients)
            std::vector<float> nabla_conv_b(num_filters, 0.0f);
            std::vector<float> nabla_conv_w(conv_weights.size(), 0.0f);
            for (int f = 0; f < num_filters; ++f) {
                for (int j = 0; j < conv_output_length; ++j) {
                    float error_signal = delta_post_conv[j * num_filters + f];
                    nabla_conv_b[f] += error_signal;
                    for (int k = 0; k < filter_size; ++k) {
                        nabla_conv_w[f * filter_size + k] += input[j + k] * error_signal;
                    }
                }
            }

            // h. Update All Weights
            for (size_t l = 0; l < dense_weights.size(); ++l) {
                for (size_t j = 0; j < dense_weights[l].size(); ++j) dense_weights[l][j] -= learning_rate * nabla_dense_w[l][j];
                for (size_t j = 0; j < dense_biases[l].size(); ++j) dense_biases[l][j] -= learning_rate * nabla_dense_b[l][j];
            }
            for (size_t j = 0; j < W_q.size(); ++j) W_q[j] -= learning_rate * nabla_Wq[j];
            for (size_t j = 0; j < W_k.size(); ++j) W_k[j] -= learning_rate * nabla_Wk[j];
            for (size_t j = 0; j < conv_weights.size(); ++j) conv_weights[j] -= learning_rate * nabla_conv_w[j];
            for (size_t j = 0; j < conv_biases.size(); ++j) conv_biases[j] -= learning_rate * nabla_conv_b[j];
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

// --- Helper Implementations (from DNN.cpp) ---
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


