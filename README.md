# Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN)

This repository contains the complete implementation, theoretical analysis, and experimental evaluation of the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) and its advanced variant, the Enhanced Random Forest Fuzzy Min-Max Neural Network (ERF-FMNN).

## Table of Contents
- [Overview](#overview)
- [Theoretical Critique & Problem Analysis](#theoretical-critique--problem-analysis)
- [Architectural Implementations](#architectural-implementations)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [License](#license)

---

## Overview

Fuzzy Min-Max Neural Networks (FMNN), introduced by Patrick K. Simpson (1992), create hyperbox-based boundaries with fuzzy membership functions. The proposed RF-FMNN enhances basic FMNN by:
1. Generating bootstrap samples with replacement until training set coverage is achieved.
2. Training independent FMNN classifiers without overlap contraction.
3. Calculating arithmetic data centroids for each hyperbox based on assigned samples and bootstrap multiplicity.
4. Resolving multi-class hyperbox overlaps using Euclidean distance to data centroids.
5. Aggregating base model predictions via majority/plurality voting.

---

## Theoretical Critique & Problem Analysis

From a statistical learning theory perspective, five fundamental limitations exist in the raw RF-FMNN specification:

1. **Discontinuous Decision Metric**: When a test sample falls inside overlapping hyperboxes, Euclidean distance to centroids is used. When outside all boxes, Simpson's boundary-decay membership is used. This causes non-monotonic boundary discontinuities.
2. **Coupon Collector Stopping Dilemma**: Generating bootstrap samples until 100% of observations are covered is the Coupon Collector's problem. For datasets of size N in [150, 600], this terminates in only 4 to 7 bags, falling short of the ensemble size needed for asymptotic variance reduction, while forcefully including noise outliers.
3. **Centroid Isotropy vs. Hyperbox Anisotropy**: Standard Euclidean distance treats all features uniformly, ignoring the rectangular bounds and aspect ratios of elongated hyperboxes.
4. **Sample Support Blind Spot**: Singleton noise hyperboxes have their centroid equal to the noise coordinate. Dense clusters have centroids in the center. Query points near the edge of a dense cluster can mistakenly be assigned to an outlier centroid.
5. **Information Discard in Hard Voting**: Discarding continuous fuzzy membership values in favor of discrete integer votes removes confidence margins and Bayesian posterior evidence.

### Proposed Research Enhancements (ERF-FMNN)
To solve these limitations, we designed ERF-FMNN:
- **Random Subspace Projection**: Projecting base learners into random feature subsets of size k = round(sqrt(d)), preventing hyperbox volume collapse.
- **Dimension-Normalized Mahalanobis Metric**: Scaling centroid distance by the dimension width (w_i - v_i) of each box.
- **Support-Weighted Density Regularization**: Weighting box relevance by tanh(0.5 * |P_j|) to suppress noise singletons.
- **Multi-Scale Hyperbox Granulation**: Stochastic theta sampling across Uniform(theta_min, theta_max).
- **Soft Continuous Fuzzy Consensus**: Bayesian-style evidence aggregation across continuous fuzzy outputs.

---

## Benchmark Results

Evaluated across 5-fold stratified cross-validation on 5 diverse datasets:

| Dataset | Basic FMNN (1992) | CART Tree | Random Forest | Proposed RF-FMNN | Enhanced RF-FMNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** (d=4, C=3) | 94.00% ± 3.89% | 95.33% ± 3.40% | 95.33% ± 3.40% | 94.00% ± 4.90% | **95.33% ± 3.40%** |
| **Wine** (d=13, C=3) | 97.75% ± 2.76% | 89.89% ± 2.90% | **98.30% ± 2.28%** | 97.75% ± 2.13% | 94.94% ± 4.53% |
| **BreastCancer** (d=30, C=2) | 95.96% ± 1.42% | 91.04% ± 2.79% | 94.91% ± 2.10% | **96.66% ± 1.02%** | 95.08% ± 1.62% |
| **TwoMoons** (d=2, C=2) | 98.67% ± 1.94% | 96.67% ± 2.79% | 97.00% ± 1.25% | **98.67% ± 1.25%** | 97.33% ± 1.33% |
| **Circles** (d=2, C=2) | 98.33% ± 1.05% | 96.67% ± 2.79% | 97.67% ± 2.00% | **98.33% ± 1.49%** | 98.00% ± 1.63% |

Key takeaway: Uncontracted centroid tie-breaking significantly improves stability and accuracy on high-dimensional data, reaching 96.66% accuracy on Breast Cancer Wisconsin (outperforming Random Forest and CART).

---

## Repository Structure

```
RF-FMNN/
├── README.md                          # Architecture and benchmark overview
├── requirements.txt                   # Environment dependencies
├── run_all.py                         # One-click execution pipeline
├── figures/                           # Generated publication figures
│   ├── benchmark_accuracy.png
│   ├── benchmark_f1.png
│   └── decision_boundaries.png
├── src/
│   ├── fmnn/
│   │   ├── base_fmnn.py               # Simpson (1992) FMNN with contraction
│   │   ├── rf_fmnn.py                 # Exact proposed RF-FMNN implementation
│   │   └── erf_fmnn.py                # Enhanced RF-FMNN (Ours)
│   ├── data/
│   │   └── dataset_loader.py          # Benchmark loader and normalizer
│   └── utils/
│       └── metrics.py                 # Evaluation metrics and box counters
├── experiments/
│   ├── run_benchmark.py               # 5-fold cross-validation suite
│   ├── ablation_study.py              # Component isolation experiments
│   ├── visualize.py                   # Boundary and benchmark plotting
│   └── results/                       # CSV and JSON benchmark results
└── paper/
    └── research_paper.md              # Complete scientific research paper
```

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.10+
- Packages: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `tabulate`

### Installation
```bash
git clone https://github.com/whysooraj/RF-FMNN.git
cd RF-FMNN
pip install -r requirements.txt
```

### Run Experiments and Generate Plots
```bash
python3 run_all.py
```

### Use in Custom Code
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
