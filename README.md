# Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN)

This repository contains the implementation, theoretical analysis, and experimental evaluation of the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) compared against classic Simpson (1992) FMNN, Decision Trees (CART), and Random Forest.

## Table of Contents
- [Overview](#overview)
- [Proposed Methodology](#proposed-methodology)
- [Architecture & Workflow](#architecture--workflow)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Ray Parallel Processing & Multi-Core Scaling](#ray-parallel-processing--multi-core-scaling)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [License](#license)

---

## Overview

The Fuzzy Min-Max Neural Network (FMNN), introduced by Patrick K. Simpson (1992), builds hyperbox-based boundaries using fuzzy membership functions. The proposed RF-FMNN model improves classification performance by:
1. Generating bootstrap samples with replacement until the union of unique observation IDs covers 100% of the training dataset.
2. Training independent FMNN classifiers on each bootstrap sample without overlap contraction.
3. Calculating arithmetic data centroids for each hyperbox based on assigned samples, counting bootstrap repetitions according to multiplicity.
4. Resolving multi-class hyperbox overlaps using Euclidean distance to data-based centroids.
5. Aggregating base model predictions via majority/plurality voting.

---

## Proposed Methodology

### 1. Bootstrap Random Sampling
Multiple bootstrap samples are generated from the training dataset using sampling with replacement. Sampling continues until every original training observation is represented in at least one bootstrap sample.

### 2. Independent FMNN Training Without Contraction
Each bootstrap sample is provided to a separate FMNN model. Each model initializes and expands hyperboxes up to maximum size parameter $\theta$. Overlapping hyperboxes belonging to different classes are not contracted or altered.

### 3. Hyperbox Centroid Calculation
For every generated hyperbox, the centroid is calculated as the arithmetic mean of all training observations assigned to that box:

$$C_j = \frac{1}{|P_j|} \sum_{x \in P_j} x$$

Multiplicity of bootstrap repetitions is accounted for in the mean.

### 4. Overlap Handling Using Nearest Centroid
When a test observation falls inside multiple overlapping hyperboxes of different classes, Euclidean distance between the observation and candidate centroids is calculated:

$$c^*(x) = \arg\min_{B_j \ni x} \|x - C_j\|_2$$

The class associated with the nearest centroid is selected.

### 5. Prediction Fallback & Ensemble Voting
If the observation does not belong to any hyperbox, standard FMNN membership functions determine the prediction. All base models submit independent predictions, and final output is decided via majority/plurality voting.

---

## Empirical Benchmark Results

Evaluated via 5-fold stratified cross-validation across 5 diverse benchmark datasets:

| Dataset | Basic FMNN (Simpson 1992) | Decision Tree (CART) | Random Forest (Breiman) | Proposed RF-FMNN |
| :--- | :---: | :---: | :---: | :---: |
| **Iris** ($d=4, C=3$) | 94.00% ± 3.89% | **95.33% ± 3.40%** | **95.33% ± 3.40%** | 94.00% ± 4.90% |
| **Wine** ($d=13, C=3$) | 97.75% ± 2.76% | 89.89% ± 2.90% | **98.30% ± 2.28%** | 97.75% ± 2.13% |
| **BreastCancer** ($d=30, C=2$) | 95.96% ± 1.42% | 91.04% ± 2.79% | 94.91% ± 2.10% | **96.66% ± 1.02%** |
| **TwoMoons** ($d=2, C=2$) | **98.67% ± 1.94%** | 96.67% ± 2.79% | 97.00% ± 1.25% | **98.67% ± 1.25%** |
| **Circles** ($d=2, C=2$) | **98.33% ± 1.05%** | 96.67% ± 2.79% | 97.67% ± 2.00% | **98.33% ± 1.49%** |

### Key Benchmark Findings:
- **High-Dimensional Precision:** On 30-dimensional Breast Cancer, Proposed RF-FMNN achieves **96.66% accuracy**, outperforming classic FMNN (95.96%), Random Forest (94.91%), and CART (91.04%). Avoiding contraction artifacts preserves cluster geometry.
- **Variance Stabilization:** On Two Moons, Proposed RF-FMNN matches top accuracy (98.67%) while reducing cross-validation standard deviation from **±1.94% to ±1.25%**.
- **Detailed 6-Metric Breakdown:** See [DETAILED_METRICS_TABLES.md](experiments/results/DETAILED_METRICS_TABLES.md) for complete Accuracy, Precision, Recall, Specificity, F1, and AUC-ROC tables.
- **Feature Subspace Sampling Theory:** See [FEATURE_SAMPLING_EXAMPLE.md](FEATURE_SAMPLING_EXAMPLE.md) for full mathematical analysis and 9-feature tumor classification walkthrough.

---

## Ray Parallel Processing & Multi-Core Scaling

Standard Simpson FMNN is strictly sequential due to inter-sample contraction dependencies. In contrast, Proposed RF-FMNN models are embarrassingly parallel across bootstrap bags. Using the [Ray](https://www.ray.io/) framework, training and inference scale seamlessly across CPU cores.

Full scaling report: [RAY_PARALLEL_SCALING_REPORT.md](experiments/results/RAY_PARALLEL_SCALING_REPORT.md).

### Multi-Core Speedup on Scaled Data ($N=3000, d=20, M=32$ Estimators)

| Cores ($p$) | Train Time | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time | Test Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 277.63s | **1.00x** | 100.0% | 232.58s | **96.17%** |
| **2** | 134.14s | **2.07x** | 103.5% | 121.25s | **96.17%** |
| **4** | 74.39s | **3.73x** | 93.3% | 63.89s | **96.17%** |
| **8** | 49.65s | **5.59x** | 69.9% | 42.90s | **96.17%** |

### Why Naive Parallelization Fails on Standard FMNN
Attempting to parallelize Simpson (1992) FMNN by partitioning data shards across cores and merging hyperboxes results in severe architectural degradation:
- **Hyperbox Explosion:** Inter-shard hyperboxes cannot be cross-contracted, adding +86 redundant hyperboxes (+11.2%).
- **Accuracy Loss:** Test accuracy drops from 92.17% down to 91.50% due to unresolved inter-class overlaps.
- **RF-FMNN Advantage:** Centroid-based voting naturally eliminates the need for inter-worker contraction, preserving 100% mathematical fidelity while achieving a 5.59x wall-clock speedup.

---

## Sensitivity Analysis

Evaluation of maximum hyperbox expansion size $\theta \in [0.1, 0.4]$ across all 5 benchmark datasets:

| Dataset | RF-FMNN ($\theta = 0.1$) | RF-FMNN ($\theta = 0.2$) | RF-FMNN ($\theta = 0.3$) | RF-FMNN ($\theta = 0.4$) |
| :--- | :---: | :---: | :---: | :---: |
| **Iris** | 94.00% | 94.00% | **95.33%** | 94.00% |
| **Wine** | 95.52% | 97.75% | 97.75% | **97.76%** |
| **BreastCancer** | **96.67%** | 96.66% | 96.14% | 94.91% |
| **TwoMoons** | **98.67%** | **98.67%** | 98.00% | 88.67% |
| **Circles** | **98.67%** | 98.33% | 97.67% | 93.33% |

---

## Repository Structure

```
RF-FMNN/
├── README.md                          # Methodology and benchmark overview
├── COMPREHENSIVE_PROJECT_REPORT.md    # Master technical and scaling report
├── WORKFLOW_AND_ARCHITECTURE.md       # Step-by-step plain-English explainer
├── DIFFERENCE_NORMAL_VS_OUR_FMNN.md   # Structural and parallel comparison vs Simpson (1992)
├── FEATURE_SAMPLING_EXAMPLE.md        # Feature sampling theory and walkthrough
├── requirements.txt                   # Environment dependencies
├── run_all.py                         # One-click execution pipeline
├── figures/                           # Generated publication figures
│   ├── benchmark_accuracy.png
│   ├── benchmark_f1.png
│   ├── decision_boundaries.png
│   ├── ray_multicore_scaling.png      # Speedup, efficiency & failure mode plots
│   └── normal_vs_rf_fmnn_concept.png  # Contraction vs Centroid geometric comparison
├── src/
│   ├── fmnn/
│   │   ├── base_fmnn.py               # Classic Simpson (1992) FMNN
│   │   ├── rf_fmnn.py                 # Proposed RF-FMNN implementation
│   │   └── ray_rf_fmnn.py             # Ray-distributed parallel RF-FMNN
│   ├── data/
│   │   └── dataset_loader.py          # Benchmark loader and normalizer
│   └── utils/
│       └── metrics.py                 # Evaluation metrics and box counters
├── experiments/
│   ├── run_benchmark.py               # 5-fold cross-validation suite
│   ├── benchmark_ray_scaling.py       # 1, 2, 4, 8 cores Ray scaling suite
│   ├── plot_scaling_and_differences.py# Scaling and conceptual plotting script
│   ├── ablation_study.py              # Theta sensitivity analysis
│   ├── visualize.py                   # Boundary and benchmark plotting
│   └── results/                       # CSV, JSON, and Markdown reports
│       ├── DETAILED_METRICS_TABLES.md
│       └── RAY_PARALLEL_SCALING_REPORT.md
└── paper/
    └── research_paper.md              # Research paper
```

---

## Quickstart & Reproduction

### Installation
```bash
git clone https://github.com/whysooraj/RF-FMNN.git
cd RF-FMNN
pip install -r requirements.txt
```

### Run All Experiments
```bash
python3 run_all.py
```

### Example Usage
```python
from src.fmnn.rf_fmnn import RFFMNNClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

clf = RFFMNNClassifier(theta=0.2, gamma=1.0)
clf.fit(X, y)
predictions = clf.predict(X)
```

---

## License
MIT License.
