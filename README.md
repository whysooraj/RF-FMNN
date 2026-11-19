# Upgraded Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN)

This branch (`enhanced-rf-fmnn`) contains the production-grade, mathematically unified implementation of RF-FMNN, addressing all 5 critical theoretical limitations of the initial proposal.

## Table of Contents
- [Summary of Applied Enhancements](#summary-of-applied-enhancements)
- [Empirical Benchmark (5-Fold Cross-Validation)](#empirical-benchmark-5-fold-cross-validation)
- [Architectural Ablation Study](#architectural-ablation-study)
- [Repository Structure](#repository-structure)
- [Replication Commands](#replication-commands)
- [License](#license)

---

## Summary of Applied Enhancements

| Issue in Raw Proposal | Applied Solution in Upgraded RF-FMNN | Impact |
| :--- | :--- | :--- |
| **Coupon Collector Stopping** (~5 bags) | Fixed ensemble size $M=35$ with Out-of-Bag (OOB) score validation | Ensures variance reduction and decorrelation |
| **Metric Isotropy Bias** (spherical Euclidean distance) | Dimension-normalized Mahalanobis distance $D_{norm}(x, C_j, B_j)$ scaled by axis spans | Correctly handles rectangular hyperbox aspect ratios |
| **Boundary Discontinuity** (mismatched inside/outside metrics) | Continuous Centroid-Anchor Fuzzy Membership $\mu_j(x) = b_j(x) \cdot \exp(-\beta D_{norm})$ | Smooth, monotonic decision contours everywhere |
| **Outlier Singletons** | Support weighting factor $W_{support} = \tanh(0.5 \cdot \|P_j\|)$ | Suppresses isolated noise points from hijacking queries |
| **Discrete Plurality Voting** | Continuous Soft Fuzzy Consensus $P(c\|x) \propto \sum_m \mu_{m, c}(x)$ | Preserves Bayesian fuzzy posterior confidence margins |

---

## Empirical Benchmark (5-Fold Cross-Validation)

Comprehensive 5-fold stratified cross-validation across 5 diverse benchmark datasets:

| Dataset | Basic FMNN (Simpson 1992) | CART Tree | Random Forest (Breiman) | Raw Proposed RF-FMNN | Upgraded RF-FMNN (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** ($d=4, C=3$) | 94.00% ± 3.89% | 95.33% ± 3.40% | 95.33% ± 3.40% | 94.00% ± 4.90% | **96.67% ± 2.98%** |
| **Wine** ($d=13, C=3$) | 97.75% ± 2.76% | 89.89% ± 2.90% | **98.30% ± 2.28%** | 97.75% ± 2.13% | 97.19% ± 2.52% |
| **BreastCancer** ($d=30, C=2$) | 95.96% ± 1.42% | 91.04% ± 2.79% | 94.73% ± 2.77% | **96.66% ± 1.02%** | 95.08% ± 1.52% |
| **TwoMoons** ($d=2, C=2$) | 98.67% ± 1.94% | 96.67% ± 2.79% | 96.67% ± 1.49% | **98.67% ± 1.25%** | 98.00% ± 0.67% |
| **Circles** ($d=2, C=2$) | 98.33% ± 1.05% | 96.67% ± 2.79% | 97.67% ± 2.00% | 98.33% ± 1.49% | **99.00% ± 1.33%** |

---

## Architectural Ablation Study

Isolating each architectural component across all datasets:

| Dataset | (1) Raw Proposal (Coupon Collector, Hard Vote) | (2) Fixed M=35 (Full Features, Static Theta) | (3) Multi-Scale Theta + Full Features | (4) Full Upgraded RF-FMNN (Subspace + Unified Anchor + Soft) |
| :--- | :---: | :---: | :---: | :---: |
| **Iris** | 94.00% | 95.33% | 95.33% | **96.67%** |
| **Wine** | **97.75%** | 95.54% | 94.97% | 97.19% |
| **BreastCancer** | **96.66%** | 96.49% | 96.14% | 95.08% |
| **TwoMoons** | **98.67%** | 98.33% | 98.00% | 98.00% |
| **Circles** | 98.33% | **99.00%** | **99.00%** | **99.00%** |

---

## Repository Structure

```
RF-FMNN/ (branch: enhanced-rf-fmnn)
├── README.md                          # Branch documentation and benchmark results
├── requirements.txt                   # Dependencies
├── run_all.py                         # Unified execution pipeline
├── figures/                           # High-resolution figures
│   ├── benchmark_accuracy.png
│   ├── benchmark_f1.png
│   └── decision_boundaries.png
├── src/
│   ├── fmnn/
│   │   ├── base_fmnn.py               # Simpson (1992) FMNN
│   │   ├── raw_rf_fmnn.py             # Raw proposal implementation
│   │   └── rf_fmnn.py                 # Upgraded production RF-FMNN engine
│   ├── data/
│   │   └── dataset_loader.py          # Benchmark loader and unit-normalizer
│   └── utils/
│       └── metrics.py                 # Evaluation metrics and box counters
└── experiments/
    ├── run_benchmark.py               # 5-fold cross-validation
    ├── ablation_study.py              # Component isolation runner
    ├── visualize.py                   # Boundary and benchmark visualizer
    └── results/                       # CSV and JSON outputs
```

---

## Replication Commands

```bash
git checkout enhanced-rf-fmnn
pip install -r requirements.txt
python3 run_all.py
```

## License
MIT License.
