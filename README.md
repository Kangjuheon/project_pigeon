# PIGEON: Flying Beyond Strong Neuron Activation Coverage

This repository provides code and experimental setup for evaluating **neuron activation coverage metrics** and their correlation with robustness in neural networks. The main contribution is **PIGEON**, a statistically-grounded extension of Strong Neuron Activation Coverage (SNAC) that incorporates confidence-based activation thresholds.

## Overview

We investigate the limitations of traditional neuron coverage metrics (e.g., SNAC) and propose confidence-thresholded variants that are sensitive to the underlying activation distribution of neurons. Experiments are conducted on MNIST using LeNet models under various adversarial training regimes (FGSM, CW, hybrid).

## Repository Structure Overview

| File Name | Description |
|-----------|-------------|
| `01_data_mnist.py` | Loads and preprocesses the MNIST dataset. |
| `02_model_lenet.py` | Defines the LeNet architecture used across all experiments. |
| `03_train_lenet.py` | Trains the base LeNet model on original MNIST. |
| `06_fgsm_lenet_mnist.py` | Generates FGSM adversarial examples from the trained LeNet. |
| `08c_cw_lenet_mnist.py` | Generates CW adversarial examples from the trained LeNet. |
| `1001_retrain_lenet_fgsm.py` | Retrains LeNet using FGSM-augmented training data. |
| `1002_retrain_lenet_cw.py` | Retrains LeNet using CW-augmented training data. |
| `1003_retrain_lenet_all.py` | Retrains LeNet using combined FGSM and CW datasets. |
| `303_neuron_stats_collector_multi.py` | Collects neuron-wise statistics (mean, std, confidence bounds) from training sets for SNAC analysis. |
| `310_250616_correlation_batch_runner.py` | Runs all SNAC and robustness metric computations across 147 model-dataset combinations and generates correlation tables. |
| `correlation_heatmaps.py` | Generates and saves correlation heatmaps (Pearson, Spearman, Kendall). |
| `run_all_experiments.py` | Runs the full experimental pipeline from training to correlation analysis. |
| `requirements.txt` | Lists all required Python packages and versions. |

### Folder Descriptions

| Folder | Description |
|--------|-------------|
| `models/` | Stores pretrained and retrained LeNet model checkpoints. |
| `neuron_stats/` | Contains saved neuron activation statistics for each model. |
| `results/` | Includes coverage results and robustness metric CSVs, pngs. |
| `correlation_heatmaps/` | Contains generated correlation heatmap visualizations (e.g., `.png` files). |

## Metrics Computed

- **SNAC-origin**: Based on max activations
- **PIGEON(SNAC-95, SNAC-99, SNAC-9999)**: Based on confidence thresholds (Gaussian assumptions)
- **Robustness Metrics**:  
  - Lipschitz Constant  
  - CLEVER L1 / L2 / Linf Scores
