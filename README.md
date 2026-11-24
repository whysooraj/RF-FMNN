# Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN)

This repository contains the implementation, theoretical analysis, and experimental evaluation of the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) compared against classic Simpson (1992) FMNN, Decision Trees (CART), and Random Forest, with distributed multi-core execution powered by stateful **Ray Actors**.

## Table of Contents
- [Overview](#overview)
- [Proposed Methodology](#proposed-methodology)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Ray Parallel Processing & Multi-Core Scaling](#ray-parallel-processing--multi-core-scaling)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction](#quickstart--reproduction)
- [License](#license)

---

## Overview

The Fuzzy Min-Max Neural Network (FMNN), introduced by Patrick K. Simpson (1992), builds hyperbox-based boundaries using fuzzy membership functions. The proposed RF-FMNN model addresses contraction artifacts, input-order sensitivity, and sequential computing bottlenecks by:
1. Generating bootstrap samples with replacement (fixed ensemble size $M$ or Coupon-Collector coverage rule, $\mathbb{E}[B] \approx \ln(N) + \gamma$), with optional feature subspace sampling (`max_features='sqrt'`).
2. Training independent vectorized FMNN classifiers on each bootstrap sample without overlap contraction.
3. Calculating arithmetic data centroids via $O(d)$ running sum and sample count (`sample_sum / sample_count`), preserving exact bootstrap multiplicity without retaining raw training copies.
4. Resolving multi-class hyperbox overlaps using Euclidean distance to data-based centroids (with deterministic tie-breaking).
5. Executing bag training and batch inference in parallel across CPU cores via stateful **Ray Actors** (`FMNNWorker`) and zero-copy Plasma shared memory.

---

## Proposed Methodology

### 1. Leak-Free Fold Preprocessing & Bootstrap Sampling
Normalization (`MinMaxScaler`) is fitted strictly on the training fold inside cross-validation, clipping test observations to $[0, 1]^d$. Multiple bootstrap samples of size $N$ are drawn with replacement. Two feature modes are supported:
- **Row Bagging (`max_features=None`)**: Uses all $d$ features per learner (preserves low-dimensional manifold geometry).
- **Subspace Sampling (`max_features='sqrt'`)**: Draws $k = \max(1, \lfloor\sqrt{d}\rfloor)$ features per learner (reduces hyperbox count by ~40% on high-dimensional data).

### 2. Independent Vectorized FMNN Training Without Contraction
Each bootstrap sample is passed to an independent `SingleFMNN` learner using vectorized per-class NumPy expansion. Overlapping hyperboxes belonging to different classes are not contracted.

### 3. Hyperbox Centroid Calculation & Overlap Resolution
Each hyperbox maintains running sum and count (`sample_sum`, `sample_count`) in $O(d)$ memory:

$$C_j = \frac{\text{sample\_sum}_j}{\text{sample\_count}_j} = \frac{1}{|P_j|} \sum_{x \in P_j} x$$

When a test observation falls inside overlapping hyperboxes of multiple classes:

$$c^*(x) = \arg\min_{B_j \ni x} \|x - C_j\|_2$$

Outside all hyperboxes, standard Simpson fuzzy membership selects the fallback prediction. All base models vote via plurality consensus.

---

## Empirical Benchmark Results

Evaluated via **5-fold stratified cross-validation** with strictly in-fold `MinMaxScaler` fitting across 6 datasets ($M=30$ estimators matched across Random Forest and RF-FMNN):

| Dataset | Basic FMNN (Simpson 1992) | Decision Tree (CART) | Random Forest (Breiman) | Proposed RF-FMNN (Bagging) | Proposed RF-FMNN (Subspace sqrt) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** ($N=150, d=4$) | 93.33% +/- 3.65% | 95.33% +/- 3.40% | **95.33% +/- 3.40%** | 94.00% +/- 4.90% | 94.00% +/- 4.90% |
| **Wine** ($N=178, d=13$) | **98.89% +/- 2.22%** | 89.89% +/- 2.90% | 98.30% +/- 2.28% | 96.63% +/- 3.26% | 96.03% +/- 5.31% |
| **BreastCancer** ($N=569, d=30$) | 95.79% +/- 2.62% | 91.04% +/- 2.79% | 94.91% +/- 2.10% | **96.84% +/- 1.31%** | 96.31% +/- 1.87% |
| **TwoMoons** ($N=300, d=2$) | **99.00% +/- 1.33%** | 96.67% +/- 2.79% | 97.00% +/- 1.25% | 98.00% +/- 1.25% | 90.33% +/- 5.42% |
| **Circles** ($N=300, d=2$) | 98.33% +/- 1.05% | 96.67% +/- 2.79% | 97.67% +/- 2.00% | **98.67% +/- 1.25%** | 82.33% +/- 4.03% |
| **Digits** ($N=1797, d=64$) | 98.11% +/- 0.33% | 85.47% +/- 1.38% | 97.44% +/- 0.41% | **98.39% +/- 0.48%** | 94.44% +/- 1.17% |

### Key Benchmark Findings:
- **Statistically Significant High-Dimensional Gain**: On 64-dimensional `Digits`, Proposed RF-FMNN (Bagging) achieves **98.39% accuracy**, beating Random Forest (97.44%) by +0.95 pp with statistical significance (paired Student t-test $p = 0.0105 < 0.05$).
- **Breast Cancer Lead**: On 30-dimensional `BreastCancer`, Proposed RF-FMNN (Bagging) achieves **96.84% +/- 1.31%**, outperforming Basic FMNN (95.79%), Random Forest (94.91%), and CART (91.04%).
- **Detailed 6-Metric & Significance Breakdown**: See [DETAILED_METRICS_TABLES.md](experiments/results/DETAILED_METRICS_TABLES.md) for Accuracy, Precision, Recall, Specificity, F1, AUC-ROC, and paired t-test p-values.
- **Feature Subspace Sampling Theory**: See [FEATURE_SAMPLING_EXAMPLE.md](FEATURE_SAMPLING_EXAMPLE.md).

---

## Ray Parallel Processing & Multi-Core Scaling

Using stateful **Ray Actors** (`@ray.remote class FMNNWorker`) + zero-copy Plasma shared memory (`ray.put`) + single-thread BLAS worker isolation (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) + pre-warmed persistent actor pools (`clf.warmup()`), trained models remain pinned in worker RAM with zero model pickling between `fit()` and `predict()`.

Full scaling & pruning report: [RAY_PARALLEL_SCALING_REPORT.md](experiments/results/RAY_PARALLEL_SCALING_REPORT.md).

### Multi-Core Scaling on Synthetic-Scaled ($N=10000, d=20, M=32$, Warm Actor Pool)

| Cores ($p$) | Warmup Time | Train Time | Speedup ($S_p$) | Efficiency ($E_p$) | Inference Time | Test Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.102s | 55.437s | **1.00x** | 100.0% | 47.325s | **93.95%** |
| **2** | 3.567s | 31.076s | **1.78x** | 89.2% | 25.963s | **93.95%** |
| **4** | 4.280s | 20.003s | **2.77x** | 69.3% | 17.459s | **93.95%** |
| **8** | 5.590s | **15.680s** | **3.54x** | 44.2% | **16.484s** | **93.95%** |

*On `Digits (N=1797, d=64)`, 8-core warm Ray Actors reach **4.00x speedup** (3.328s to 0.831s) and **3.27x inference speedup** (4.350s to 1.332s).*

### Fair Comparison Against Vectorized Normal BaseFMNN ($N=10000, d=20$)
Both `BaseFMNN` and `RF-FMNN` are vectorized in NumPy for apples-to-apples timing:
- **1-Core Vectorized BaseFMNN**: 4.917s train time, **90.00%** accuracy, 2325 hyperboxes.
- **8-Core Naive Box-Pooling BaseFMNN**: 6.515s (0.75x slowdown), **90.20%** accuracy, 2604 hyperboxes (+12.0% box bloat).
- **8-Core Fair Sharded-Voting BaseFMNN**: 0.470s (each worker sees $N/8 = 1000$ rows), **90.90%** accuracy.
- **8-Core Ray Actor RF-FMNN ($M=32$)**: **15.680s** (3.54x vs serial RF-FMNN), **93.95% accuracy (+3.95 pp over 1-core BaseFMNN, +3.05 pp over Sharded-Voting BaseFMNN)**; $M=64$ reaches **94.35% (+4.35 pp)** in 25.23s.

---

## Sensitivity Analysis

Leak-free 5-fold CV evaluation of maximum hyperbox expansion size $\theta \in [0.1, 0.4]$ (`n_estimators=30`):

| Dataset | RF-FMNN ($\theta = 0.1$) | RF-FMNN ($\theta = 0.2$) | RF-FMNN ($\theta = 0.3$) | RF-FMNN ($\theta = 0.4$) |
| :--- | :---: | :---: | :---: | :---: |
| **Iris** | 94.67% | 94.00% | 96.00% | **96.67%** |
| **Wine** | 94.97% | 96.63% | **98.30%** | 97.76% |
| **BreastCancer** | **97.01%** | 96.84% | 95.79% | 95.44% |
| **TwoMoons** | **99.33%** | 98.00% | 97.67% | 90.67% |
| **Circles** | **98.67%** | **98.67%** | 98.33% | 95.67% |
| **Digits** | 98.11% | **98.39%** | 98.11% | 97.61% |

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
│   ├── ray_multicore_scaling.png      # Ray Actor scaling & BaseFMNN sharding plots
│   └── normal_vs_rf_fmnn_concept.png  # Contraction vs Centroid geometric comparison
├── src/
│   ├── fmnn/
│   │   ├── base_fmnn.py               # Classic Simpson (1992) FMNN
│   │   ├── rf_fmnn.py                 # Vectorized RF-FMNN (Bagging + Subspace sqrt)
│   │   └── ray_rf_fmnn.py             # Stateful Ray Actor RF-FMNN
│   ├── data/
│   │   └── dataset_loader.py          # Raw leak-free dataset loader (6 datasets incl. Digits)
│   └── utils/
│       └── metrics.py                 # 6-metric evaluation suite
├── experiments/
│   ├── run_benchmark.py               # Leak-free 5-fold CV + paired t-test suite
│   ├── benchmark_ray_scaling.py       # 1, 2, 4, 8 cores Ray Actor scaling suite
│   ├── plot_scaling_and_differences.py# Scaling and conceptual plotting script
│   ├── ablation_study.py              # Leak-free theta sensitivity analysis
│   ├── visualize.py                   # Boundary and benchmark plotting
│   └── results/                       # CSV, JSON, and Markdown reports
│       ├── DETAILED_METRICS_TABLES.md
│       ├── RAY_PARALLEL_SCALING_REPORT.md
│       └── statistical_significance.json
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

### Run Benchmarks & Ray Scaling
```bash
PYTHONPATH=. python3 experiments/run_benchmark.py
PYTHONPATH=. python3 experiments/ablation_study.py
PYTHONPATH=. python3 experiments/visualize.py
PYTHONPATH=. python3 experiments/benchmark_ray_scaling.py
PYTHONPATH=. python3 experiments/plot_scaling_and_differences.py
```

### Example Usage (Stateful Ray Actors)
```python
from src.fmnn.ray_rf_fmnn import RayRFFMNNClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = np.clip(scaler.transform(X_te), 0.0, 1.0)

clf = RayRFFMNNClassifier(theta=0.2, gamma=1.0, n_estimators=32, num_cpus=4)
clf.fit(X_tr, y_tr)
predictions = clf.predict(X_te)
clf.close()
```

---

## License
MIT License.
