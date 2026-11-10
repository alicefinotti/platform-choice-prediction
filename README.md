# Customer Platform Choice Prediction

## Overview

A shared-encoder neural network for predicting customers’ choice between three online shopping platforms based on **delivery time, cost, customer group, and membership status**. The analysis uses **7,000 choice scenarios**, with the task formulated as three-class classification.

## Approach

- **Exploratory analysis:** Summary statistics and comparisons of delivery time and cost distributions across platforms.
- **Feature representation:** Convert each choice scenario into a **3 × 4 tensor** containing two customer attributes and two platform-specific attributes for each alternative.
- **Benchmark:** Multinomial logistic regression on the flattened features.
- **Neural network:** Shared multilayer perceptron (4 → 32 → 16 → 1) produces one score per platform, with cross-entropy loss.
- **Training and evaluation:** Adam hyperparameter search, early stopping, a separate validation set, and held-out test evaluation. A platform-swap test verifies **permutation equivariance**.

## Results

The original project reported a **66.9% in-sample accuracy** for the logistic regression benchmark and approximately **70% validation accuracy** for the shared-encoder network. The notebook computes comparable validation and held-out test metrics for both models using a consistent data split.
