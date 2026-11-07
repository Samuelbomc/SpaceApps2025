# Kepler's Algorithm

> **AI-use disclaimer:** This project was developed with assistance from AI tools. AI-generated suggestions were reviewed and adapted by the project team, the team remains responsible for the implementation and its results.

Kepler's Algorithm is a C++20 machine-learning prototype for classifying astronomical observations as likely planets or not.

For more information about the project, visit the [Space Apps Challenge project page](https://www.spaceappschallenge.org/2025/find-a-team/keplers-algorithm/?tab=project).

## What it offers

- **Two-class classification:** predicts whether an observation is a confirmed/known planet candidate or a false positive.
- **Multiple datasets:** includes loading and preprocessing workflows for K2, TESS, and Kepler CSV data.
- **Data preparation:** filters supported labels, shuffles the observations, creates an 80/20 training/testing split, and normalizes numeric features using statistics calculated from the training set.
- **Hybrid neural network:** combines a one-dimensional convolutional layer, max pooling, dense layers, and a self-attention gate implemented directly in C++.
- **Training and evaluation:** trains the model with cross-entropy loss and reports the final accuracy on the held-out test data.

## Current status

The current executable runs the K2 training workflow by default and prints training progress, sample counts, and testing accuracy to the console. The TESS and Kepler workflows are implemented in the code and can be selected by calling `train_model("TESS")` or `train_model("Kepler")` from `main/SpaceApps2025.cpp`.

The web visualization is not implemented.

## Building

Requirements:

- CMake 3.31 or newer
- A C++20-compatible compiler
- Ninja (or another supported CMake generator)
- Crow, available through the configured CMake package setup

Configure and build from the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

The CSV datasets expected by the program are located in the `data/` directory. After building, run the generated `SpaceApps2025` executable from an environment where the configured source path and dataset files are available.

## Project layout

- `main/` — application entry point and dataset training workflows
- `data/` — CSV loading, filtering, splitting, and normalization
- `models/` — neural-network and training implementation
- `web/` — Crow server and API prototype
- `html/` — browser interface prototype
