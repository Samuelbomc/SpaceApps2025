#pragma once
#include <vector>

struct DatasetSplit {
    std::vector<std::vector<float>> training_inputs;
    std::vector<std::vector<float>> training_targets;
    std::vector<std::vector<float>> testing_inputs;
    std::vector<std::vector<float>> testing_targets;
};