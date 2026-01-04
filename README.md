# Higgs Boson Classification: A Comparative Study of Machine Learning Methods

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Authors

- **Tolga Kuntman**
- **Michel Moussally**

## Abstract

High-energy physics experiments produce extremely large datasets in which events of physical interest are difficult to distinguish from background noise. Effectively addressing this challenge requires machine learning models that can capture complex patterns in high-dimensional data.

In this project, we investigate a binary classification problem using the **HIGGS dataset**—a widely used benchmark composed of simulated proton–proton collision events represented by 28 numerical features. We perform an independent implementation and methodological comparison of three representative machine learning model families:

- **Logistic Regression** — a linear baseline
- **Gradient-Boosted Decision Trees** — a nonlinear ensemble method
- **Feed-Forward Neural Network (MLP)** — a deep learning approach

A balanced subset of 500,000 events is sampled from the original dataset of approximately 11 million events to reduce computational complexity. All models are trained and evaluated using a consistent training, validation, and test pipeline, with performance primarily assessed using the **ROC-AUC score**.

## Documentation

- 📄 **[Project Report](Report_Weiyi_C_KUNTMAN_MOUSSALLY.pdf)**
- 🖼️ **[Project Poster](Poster_Weiyi_C_Kuntman_Moussally.pdf)**

## Dataset

### Overview

We use the **HIGGS dataset**, a large-scale benchmark dataset for studying machine learning methods in high-energy physics. The dataset is publicly available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS) and can be freely accessed for research and educational purposes.

The dataset was generated using detailed Monte Carlo simulations of proton-proton collisions at the Large Hadron Collider (LHC), with the goal of distinguishing **Higgs boson signal events** from **Standard Model background events**.

### Task Description

- **Type**: Binary classification
- **Objective**: Predict whether an event corresponds to a Higgs boson signal (label = 1) or background noise (label = 0)
- **Classes**: Approximately balanced — 52.9% signal events, 47.1% background events

### Features

Each event is described by **28 continuous numerical features**:
- **21 raw kinematic measurements** (e.g., particle momentum and angular variables)
- **7 engineered physics features** (e.g., invariant masses, angular separations, and event-level quantities)

### Data Sampling

From the full dataset of ~11 million events, we randomly sampled **500,000 events** while preserving the original class ratio (52.9% signal / 47.1% background). This allows us to balance computational feasibility with statistical representativeness.

### Data Split

The sampled dataset is split into training, validation, and test sets using an **80/10/10 split**:

| Split | Size | Purpose |
|-------|------|---------|
| Training | 400,000 | Model fitting |
| Validation | 50,000 | Hyperparameter tuning and model selection |
| Test | 50,000 | Final performance evaluation |

Class proportions remain nearly identical across all splits.

## Project Structure

```
machine_learning_project/
├── data/
│   ├── HIGGS.csv                    # Full dataset (download separately)
│   └── samples/
│       ├── higgs_train.csv          # Training set (400K samples)
│       ├── higgs_val.csv            # Validation set (50K samples)
│       └── higgs_test.csv           # Test set (50K samples)
├── notebooks/
│   ├── data_analysis.ipynb          # Exploratory data analysis
│   ├── logistic_regression.ipynb    # Logistic regression implementation
│   ├── gradient_boosted_tree.ipynb  # Gradient boosted trees implementation
│   └── feed_forward_MLP.ipynb       # Neural network (MLP) implementation
├── src/
│   ├── utils.py                     # Utility functions (data loading, plotting)
│   ├── public_tests.py              # Test functions from lab assignments
│   └── test_utils.py                # Testing utilities from lab assignments
├── environment.yml                  # Conda environment specification
├── Report_Weiyi_C_KUNTMAN_MOUSSALLY.pdf
├── Poster_Weiyi_C_Kuntman_Moussally.pdf
└── README.md
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| **[data_analysis.ipynb](notebooks/data_analysis.ipynb)** | Exploratory data analysis including feature distributions, correlations, and class balance visualization |
| **[logistic_regression.ipynb](notebooks/logistic_regression.ipynb)** | Implementation of logistic regression from scratch using gradient descent as a linear baseline model |
| **[gradient_boosted_tree.ipynb](notebooks/gradient_boosted_tree.ipynb)** | Gradient boosted decision trees using scikit-learn with hyperparameter tuning |
| **[feed_forward_MLP.ipynb](notebooks/feed_forward_MLP.ipynb)** | Feed-forward neural network (Multi-Layer Perceptron) implemented in PyTorch with hyperparameter search |

## Getting Started

### Installation

1. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**
   ```bash
   conda activate ml_project
   ```

3. **Download the HIGGS dataset**
   
   Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS) and place `HIGGS.csv` in the `data/` directory.
   
   Alternatively, the sampled splits (`higgs_train.csv`, `higgs_val.csv`, `higgs_test.csv`) are provided in `data/samples/` for convenience.

## Reproducing Results

To reproduce our results:

1. **Set up the environment** as described above

2. **Run the notebooks in order**:
   - Start with `data_analysis.ipynb` to understand the dataset
   - Run `logistic_regression.ipynb` for the linear baseline
   - Run `gradient_boosted_tree.ipynb` for the ensemble method
   - Run `feed_forward_MLP.ipynb` for the neural network

3. **Evaluation metric**: All models are evaluated using the **ROC-AUC score** on the held-out test set

> **Note**: The MLP hyperparameter search may take significant time depending on your hardware. GPU acceleration is supported if CUDA is available.

## Key Results

| Model | Test ROC-AUC |
|-------|--------------|
| Logistic Regression | ~0.69 |
| Gradient Boosted Trees | ~0.83 |
| Feed-Forward MLP | ~0.84 |

The results demonstrate that nonlinear models significantly outperform the linear baseline, highlighting the importance of modeling nonlinear feature interactions in high-energy physics classification tasks.

## Acknowledgments

- Parts of the logistic regression implementation are based on lab assignments (marked with comments in the code)
- HIGGS dataset provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)
- Original dataset paper: Baldi, P., Sadowski, P., & Whiteson, D. (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
