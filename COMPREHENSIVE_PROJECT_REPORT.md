# Comprehensive Technical Report: RF-FMNN Architecture, Multi-Metric Benchmarks, and Multi-Core Ray Scalability Analysis

---

## 1. Executive Summary & Project Objective

The Fuzzy Min-Max Neural Network (FMNN), introduced by Patrick K. Simpson in 1992, is an online neuro-fuzzy architecture that learns pattern clusters via $n$-dimensional hyperboxes. While interpretable and fast for small streams, classic FMNN exhibits three structural bottlenecks:
1. **Destructive Overlap Contraction**: When hyperboxes of different classes collide, a 4-case heuristic clips coordinate boundaries along the dimension of minimal overlap, distorting natural cluster geometry.
2. **Severe Input-Order Sensitivity**: Because contraction depends on sequential pairwise collisions, changing the presentation order of training vectors reshapes the hyperbox configuration.
3. **Inherent Sequentiality (Parallelization Bottleneck)**: Standard online FMNN cannot be parallelized across multi-core processors without altering the algorithm. Sharding training data across cores either breaks global contraction (box-pooling collisions + hyperbox bloat) or shrinks per-learner sample coverage to $N/p$ (degrading test accuracy even under majority voting).

**Project Objective**:
Implement, benchmark, and evaluate the **Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN)** using strictly leak-free cross-validation and stateful **Ray Actor** parallelization:
- Replacing sequential contraction with an independent bootstrap ensemble (`n_estimators=30` matched to Random Forest, or Coupon-Collector stopping rule).
- Supporting both full-feature row bagging (`max_features=None`) and feature-subspace sampling (`max_features='sqrt'`).
- Maintaining arithmetic data centroids via running sum and sample count (`sample_sum / sample_count`) in $O(d)$ memory (zero raw sample copy retention, exact bootstrap multiplicity).
- Vectorizing both `SingleFMNN.fit()` and batch `predict()` in C-level NumPy operations.
- Distributing training and inference across 1, 2, 4, and 8 CPU cores using **Stateful Ray Actors** (`@ray.remote class FMNNWorker`) + zero-copy Plasma Object Store (`ray.put`).

---

## 2. Architecture & Algorithmic Workflow

### 2.1 The RF-FMNN Pipeline

```
+-------------------------------------------------------------------------+
|                    Raw Fold Training Data (N_tr x d)                    |
|         -> Fit MinMaxScaler strictly on X_train (test clipped [0,1])    |
+-------------------------------------------------------------------------+
                                     |
             [ Bootstrap Sampling (Fixed M or Coupon-Collector) ]
              Optional feature subspace selection: k = floor(sqrt(d))
                                     |
             [ ray.put(X_train), ray.put(y_train) -> Plasma Store ]
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
   FMNNWorker Actor 1          FMNNWorker Actor 2          FMNNWorker Actor p
   (Pinned Worker RAM)         (Pinned Worker RAM)         (Pinned Worker RAM)
         |                           |                           |
  [ Vectorized Expand ]       [ Vectorized Expand ]       [ Vectorized Expand ]
   - Bound: theta              - Bound: theta              - Bound: theta
   - NO CONTRACTION            - NO CONTRACTION            - NO CONTRACTION
   - O(d) sample_sum/count     - O(d) sample_sum/count     - O(d) sample_sum/count
         |                           |                           |
   Models stay pinned          Models stay pinned          Models stay pinned
   (Zero model pickling)       (Zero model pickling)       (Zero model pickling)
         \                           |                           /
          +--------------------------+--------------------------+
                                     |
                     Test Query Batch X_test -> Plasma Store
                                     |
           +-------------------------+-------------------------+
           |                                                   |
   [ Contained in Box(es)? ]                           [ Outside All Boxes? ]
           |                                                   |
    Overlap detected?                                   Evaluate FMNN
   /                 \                                  Fuzzy Membership
 Single Class    Multi-Class                                   |
      |               |                               Select Argmax Mem
 Output Class    Argmin ||x - C_j||_2                          |
      \               /                                        |
       +------+------+                                         |
              |                                                |
    [ Actor Returns (N_test x C) Vote Matrix ] <---------------+
              |
    [ Driver Sums Vote Matrices -> Deterministic Argmax ] -> Final Prediction
```

### 2.2 Mathematical Specifications

#### A. Bootstrap Sample Budget & Coupon-Collector Formula
For a training dataset of $N$ observations, each bag draws $N$ instances with replacement. The asymptotic out-of-bag fraction is:
$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679$$
Each bag contains $\approx 63.2\%$ unique observations. Under the Coupon-Collector stopping rule, bags are drawn iteratively until the cumulative union covers 100% of observation IDs:
$$\bigcup_{b=1}^{B} S_b = \{1, 2, \dots, N\}$$
Because each bag draws $N$ samples (while total individual sample draws required is $\approx N(\ln N + \gamma)$), the **expected number of bags** is:
$$\mathbb{E}[B] \approx \ln(N) + \gamma, \quad \gamma \approx 0.5772 \text{ (Euler-Mascheroni constant)}$$
For $N \in [150, 1000]$, $\mathbb{E}[B] \approx 5.6 \text{ to } 7.5$ bags. For fair variance stabilization against Breiman's Random Forest in cross-validation benchmarks, we also fix $M = 30$ estimators (`n_estimators=30`).

#### B. Feature Subspace Sampling (`max_features='sqrt'`)
When feature subspace mode is enabled:
- Number of selected features per learner: $k = \max(1, \lfloor\sqrt{d}\rfloor)$.
- Probability a single feature is excluded across all $M$ learners:
  $$P(\text{Feature } j \text{ absent everywhere}) = \left(1 - \frac{k}{d}\right)^M$$
  For $d=30, k=5, M=35$: $(25/30)^{35} = (5/6)^{35} \approx 0.0017$ (0.17%).

#### C. Hyperbox Expansion Criterion
A hyperbox $B_j = (V_j, W_j)$ expands to absorb vector $x$ if the maximum size bound $\theta \in (0, 1]$ is satisfied:
$$\sum_{i=1}^{d} \left[\max(w_{ji}, x_i) - \min(v_{ji}, x_i)\right] \le d \cdot \theta$$

#### D. Overlap Resolution via $O(d)$ Multiplicity Centroids
Hyperboxes are never contracted. Each box maintains running `sample_sum` and `sample_count` (zero raw sample arrays retained):
$$C_j = \frac{\text{sample\_sum}_j}{\text{sample\_count}_j} = \frac{1}{|P_j|} \sum_{x \in P_j} x$$
If query $x$ falls inside overlapping boxes belonging to multiple distinct classes $\mathcal{O}(x)$:
$$c^*(x) = \arg\min_{B_j \in \mathcal{O}(x)} \|x - C_j\|_2$$
With deterministic tie-breaking by lowest class index. Outside all hyperboxes, standard Simpson fuzzy membership selects the class:
$$b_j(x) = \frac{1}{2d} \sum_{i=1}^{d} \left[ \max\left(0, 1 - \max(0, \gamma \min(1, x_i - w_{ji}))\right) + \max\left(0, 1 - \max(0, \gamma \min(1, v_{ji} - x_i))\right) \right]$$

---

## 3. Leak-Free Multi-Metric Benchmark Results (6 Datasets, 5 Models)

Evaluated via **5-Fold Stratified Cross-Validation** with `MinMaxScaler` fitted strictly on each training fold (`X_test` clipped to `[0, 1]`).
Ensemble budgets matched at $M=30$ (`n_estimators=30`) for Random Forest and both Proposed RF-FMNN modes.
Full per-model 6-metric tables in [DETAILED_METRICS_TABLES.md](experiments/results/DETAILED_METRICS_TABLES.md).

### 3.1 Overall Comparative Accuracy & F1 Summary

| Dataset | Metric | Basic FMNN (1992) | CART Tree | Random Forest | Proposed RF-FMNN (Bagging) | Proposed RF-FMNN (Subspace sqrt) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** ($N=150, d=4$) | Accuracy | 93.33% +/- 3.65% | 95.33% +/- 3.40% | **95.33% +/- 3.40%** | 94.00% +/- 4.90% | 94.00% +/- 4.90% |
| | F1-Score | 93.28% +/- 3.70% | 95.31% +/- 3.41% | **95.31% +/- 3.41%** | 93.96% +/- 4.94% | 93.96% +/- 4.94% |
| **Wine** ($N=178, d=13$) | Accuracy | **98.89% +/- 2.22%** | 89.89% +/- 2.90% | 98.30% +/- 2.28% | 96.63% +/- 3.26% | 96.03% +/- 5.31% |
| | F1-Score | **98.90% +/- 2.19%** | 90.12% +/- 2.80% | 98.40% +/- 2.09% | 96.66% +/- 3.19% | 96.16% +/- 5.02% |
| **BreastCancer** ($N=569, d=30$) | Accuracy | 95.79% +/- 2.62% | 91.04% +/- 2.79% | 94.91% +/- 2.10% | **96.84% +/- 1.31%** | 96.31% +/- 1.87% |
| | F1-Score | 95.45% +/- 2.82% | 90.28% +/- 3.17% | 94.50% +/- 2.34% | **96.61% +/- 1.41%** | 96.07% +/- 1.95% |
| **TwoMoons** ($N=300, d=2$) | Accuracy | **99.00% +/- 1.33%** | 96.67% +/- 2.79% | 97.00% +/- 1.25% | 98.00% +/- 1.25% | 90.33% +/- 5.42% |
| | F1-Score | **99.00% +/- 1.33%** | 96.66% +/- 2.79% | 97.00% +/- 1.25% | 98.00% +/- 1.25% | 90.25% +/- 5.55% |
| **Circles** ($N=300, d=2$) | Accuracy | 98.33% +/- 1.05% | 96.67% +/- 2.79% | 97.67% +/- 2.00% | **98.67% +/- 1.25%** | 82.33% +/- 4.03% |
| | F1-Score | 98.33% +/- 1.06% | 96.66% +/- 2.80% | 97.66% +/- 2.00% | **98.67% +/- 1.25%** | 81.84% +/- 4.38% |
| **Digits** ($N=1797, d=64$) | Accuracy | 98.11% +/- 0.33% | 85.47% +/- 1.38% | 97.44% +/- 0.41% | **98.39% +/- 0.48%** | 94.44% +/- 1.17% |
| | F1-Score | 98.10% +/- 0.33% | 85.48% +/- 1.40% | 97.43% +/- 0.41% | **98.37% +/- 0.49%** | 94.34% +/- 1.22% |

### 3.2 Statistical Significance (Paired Student t-test, $\alpha = 0.05$)

Key findings from two-sided paired t-tests across 5 CV folds ([DETAILED_METRICS_TABLES.md](experiments/results/DETAILED_METRICS_TABLES.md)):
- **Digits ($d=64, C=10$)**: Proposed RF-FMNN (Bagging) achieves **98.39%**, outperforming Random Forest (97.44%) by **+0.95 pp** with **statistical significance ($t = +4.545, p = 0.0105 < 0.05$)**.
- **BreastCancer ($d=30, C=2$)**: Proposed RF-FMNN (Bagging) achieves highest overall accuracy (**96.84% +/- 1.31%**), outperforming Random Forest by **+1.93 pp** ($p = 0.0854$) and Basic FMNN by **+1.05 pp** ($p = 0.2835$) while halving fold standard deviation vs Basic FMNN (2.62% to 1.31%).
- **Row Bagging vs Subspace Sampling**: On low-dimensional non-linear manifolds (`TwoMoons` and `Circles`, $d=2 \implies k=1$), dropping one axis breaks 2D manifold geometry (statistically significant degradation, $p = 0.0010$ on Circles). On high-dimensional `BreastCancer` ($d=30$), subspace sampling reduces hyperbox count by **39.8%** (1740.4 to 1048.0 boxes) while retaining 96.31% accuracy.

---

## 4. Multi-Core Ray Actor Scaling & Speedup Analysis

Evaluated on a 12-core Linux host with Ray 2.60.0 using **Stateful Ray Actors** (`FMNNWorker`), zero-copy Plasma Object Store, vectorized `SingleFMNN.fit()`/`predict()`, and strictly train-only MinMax scaling.
Full report: [RAY_PARALLEL_SCALING_REPORT.md](experiments/results/RAY_PARALLEL_SCALING_REPORT.md) | Plot: `figures/ray_multicore_scaling.png`.

### 4.1 RF-FMNN Ray Actor Scaling ($M=32$ Estimators)

#### Synthetic-Scaled ($N=5000, d=20$)
| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.549s | **1.00x** | 100.0% | 9.281s | **94.90%** | 25597 |
| **2** | 10.290s | **1.41x** | 70.7% | 5.751s | **94.90%** | 25597 |
| **4** | 6.766s | **2.15x** | 53.8% | 3.462s | **94.90%** | 25597 |
| **8** | **6.709s** | **2.17x** | 27.1% | **2.923s** | **94.90%** | 25597 |

#### Real High-Dimensional: Digits ($N=1797, d=64$)
| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.095s | **1.00x** | 100.0% | 3.197s | **97.78%** | 10059 |
| **2** | 3.229s | **1.27x** | 63.4% | 2.339s | **97.78%** | 10059 |
| **4** | 2.903s | **1.41x** | 35.3% | 1.362s | **97.78%** | 10059 |
| **8** | 3.513s | **1.17x** | 14.6% | **1.104s** | **97.78%** | 10059 |

#### Small Dataset: BreastCancer ($N=569, d=30$)
| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.472s | **1.00x** | 100.0% | 0.188s | **96.49%** | 1828 |
| **2** | 2.479s | **1.00x** | 49.9% | 0.118s | **96.49%** | 1828 |
| **4** | 2.290s | **1.08x** | 27.0% | 0.084s | **96.49%** | 1828 |
| **8** | 3.212s | **0.77x** | 9.6% | **0.067s** | **96.49%** | 1828 |

**Key Systems Findings**:
- **Vectorization Breakthrough**: Vectorizing `SingleFMNN.fit()` and removing `box.samples` memory retention slashed 1-core RF-FMNN training ($M=32$) from >340s (unvectorized on $N=3000$) down to **14.55s on larger $N=5000$** — making **8-core Ray Actor RF-FMNN (6.71s) 4.18x faster than 1-core Normal BaseFMNN (28.03s)** while achieving **+2.40 pp higher test accuracy (94.90% vs 92.50%)**!
- **Amdahl's Law & Actor Startup Floor**: Because core training is now sub-second per bag, spawning 8 fresh Ray Actor processes inside `fit()` adds a ~2.5s fixed startup floor. On small datasets (`BreastCancer`, 2.47s on 1 core), 8-core spawn cost exceeds compute savings (`0.77x`). On larger ensembles (`M=64` on `Synthetic-Scaled`), actor startup amortizes cleanly, driving throughput from **2.07 bags/sec ($M=8$) to 5.52 bags/sec ($M=64$)** in 11.59s!
- **Zero-Pickling Inference**: Keeping models pinned inside `FMNNWorker` Actor RAM eliminates model serialization between `fit()` and `predict()`, cutting inference latency to **0.067s** on BreastCancer, **1.10s** on Digits, and **2.92s** on Synthetic-Scaled.

### 4.2 Estimator Depth Scaling on 8 Cores ($N=5000, d=20$)
| Estimators ($M$) | Train Time (s) | Throughput (Bags/sec) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: |
| **8** | 3.856s | 2.07 | 93.70% | 6372 |
| **16** | 5.885s | 2.72 | 95.50% | 12752 |
| **32** | 7.578s | 4.22 | 94.90% | 25597 |
| **64** | 11.592s | **5.52** | **95.00%** | 51322 |

---

## 5. Parallelizing Normal FMNN: Naive Box Pooling vs Fair Sharded Voting

Evaluated on `Synthetic-Scaled (N=5000, d=20)` (4000 train, 1000 test) comparing two ways of parallelizing Simpson (1992) BaseFMNN across $p$ cores:

### 5.1 Naive Data-Partitioned Box Pooling
| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Total Hyperboxes | Box Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4000 | 28.034s | 1.00x | **92.50%** | 1235 | Baseline |
| **2** | 2000 | 9.982s | 2.81x | **90.70%** | 1269 | +34 |
| **4** | 1000 | 4.519s | 6.20x | **91.70%** | 1316 | +81 |
| **8** | 500 | 3.978s | 7.05x | **90.80%** | 1367 | +132 (+10.7%) |

### 5.2 Fair Sharded Majority-Voting BaseFMNN
| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Acc Delta vs 1-Core | Total Hyperboxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4000 | 28.034s | 1.00x | **92.50%** | Baseline | 1235 |
| **2** | 2000 | 8.368s | 3.35x | **89.90%** | -2.60 pp | 1269 |
| **4** | 1000 | 2.677s | 10.47x | **90.70%** | -1.80 pp | 1316 |
| **8** | 500 | 1.044s | 26.84x | **91.80%** | -0.70 pp | 1367 |

**Architectural Takeaway**:
- **Naive Box Pooling** fails because cross-shard overlapping boxes are never contracted, dropping accuracy by **-1.70 pp** (92.50% to 90.80%) and bloating hyperbox counts by **+10.7%** (+132 boxes).
- **Fair Sharded Voting** avoids cross-shard collisions and runs rapidly because each worker processes only $N/p = 500$ samples, **but** reducing per-learner training data by $8\times$ degrades accuracy (-0.70 pp to -2.60 pp below 1-core BaseFMNN).
- **Proposed RF-FMNN** trains each learner on full $N$-size bootstrap bags across Ray Actors, achieving **94.90% accuracy (+3.10 pp above Sharded-Voting BaseFMNN and +2.40 pp above 1-core BaseFMNN)** while still training **4.18x faster than 1-core BaseFMNN (6.71s vs 28.03s)**.

---

## 6. Honest Technical Critique & Academic Framing

### 6.1 Verified Strengths
1. **Faster AND More Accurate Than 1-Core Normal FMNN at Scale**: On $N=5000$, 8-core Stateful Ray Actor RF-FMNN trains in **6.71s** (4.18x faster than 1-core BaseFMNN's 28.03s) and gains **+2.40 pp** accuracy (94.90% vs 92.50%).
2. **Statistically Significant High-Dimensional Performance**: On 64-dimensional `Digits`, RF-FMNN (Bagging) achieves **98.39%** accuracy, beating Random Forest (97.44%) with $p = 0.0105$. On 30-dimensional `BreastCancer`, RF-FMNN leads all models at **96.84%**.
3. **Zero-Leakage Rigor & Clean Systems Engineering**: Strict in-fold MinMax scaling, $O(d)$ running-sum centroids, vectorized `fit()`/`predict()`, stateful Ray Actors with zero model pickling.

### 6.2 Honest Limitations
1. **Accuracy Parity on Small Toy Datasets**: On `Iris`, `Wine`, `TwoMoons`, and `Circles`, differences between Basic FMNN and RF-FMNN (Bagging) are **not statistically significant** ($p > 0.10$).
2. **Hyperbox Memory Footprint**: Storing $M=30$ bags generates $20\times\text{--}25\times$ more hyperboxes than single BaseFMNN (e.g., 9441.6 vs 408.8 on Digits). Enabling `max_features='sqrt'` cuts box count by ~40% on high-$d$ data at a modest accuracy trade-off.
3. **Ray Actor Startup Overhead on Tiny Data**: Spawning 8 Ray Actor OS processes takes ~2.5s; parallelizing tiny datasets ($N < 1000$) yields `<1x` speedup. Use Ray Actors when $N \ge 2000$ or $M \ge 32$.

---

## 7. Repository Layout & Artifacts

```
RF-FMNN/
├── README.md                          # Project overview and quickstart
├── COMPREHENSIVE_PROJECT_REPORT.md    # Master technical report (this document)
├── DIFFERENCE_NORMAL_VS_OUR_FMNN.md   # Structural and parallel comparison vs Simpson (1992)
├── FEATURE_SAMPLING_EXAMPLE.md        # Feature sampling theory and 9-feature walkthrough
├── WORKFLOW_AND_ARCHITECTURE.md       # Plain-English workflow explainer
├── requirements.txt                   # Dependencies (NumPy, SciPy, scikit-learn, Ray)
├── run_all.py                         # Complete pipeline runner
├── figures/                           # Publication figures
│   ├── benchmark_accuracy.png         # 5-model cross-validation accuracy bars
│   ├── benchmark_f1.png               # 5-model cross-validation F1-score bars
│   ├── decision_boundaries.png        # 2D non-linear decision boundary comparison
│   ├── ray_multicore_scaling.png      # 4-panel Ray Actor scaling & BaseFMNN sharding plots
│   └── normal_vs_rf_fmnn_concept.png  # Simpson contraction vs centroid resolution diagram
├── src/
│   ├── fmnn/
│   │   ├── base_fmnn.py               # Classic Simpson (1992) FMNN
│   │   ├── rf_fmnn.py                 # Vectorized RF-FMNN (Bagging + Subspace sqrt)
│   │   └── ray_rf_fmnn.py             # Stateful Ray Actor FMNNWorker implementation
│   ├── data/
│   │   └── dataset_loader.py          # Raw leak-free dataset loader (6 datasets incl. Digits)
│   └── utils/
│       └── metrics.py                 # 6-metric evaluation suite
└── experiments/
    ├── run_benchmark.py               # Leak-free 5-fold CV + paired t-test suite
    ├── benchmark_ray_scaling.py       # 1, 2, 4, 8 core Ray Actor & sharding suite
    ├── plot_scaling_and_differences.py# Ray scaling and conceptual diagram plotter
    ├── ablation_study.py              # Leak-free theta sensitivity analysis
    ├── visualize.py                   # Benchmark bar & decision boundary plotter
    └── results/                       # Experimental data artifacts
        ├── benchmark_results.csv
        ├── benchmark_results.json
        ├── statistical_significance.json
        ├── ablation_results.csv
        ├── ablation_results.json
        ├── DETAILED_METRICS_TABLES.md
        ├── ray_scaling_results.json
        └── RAY_PARALLEL_SCALING_REPORT.md
```
