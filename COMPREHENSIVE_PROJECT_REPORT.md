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

Evaluated on a 12-thread Linux host (Intel hybrid 4 P-cores + 4 E-cores) with Ray 2.60.0 using **Stateful Ray Actors** (`FMNNWorker`), zero-copy Plasma Object Store, **single-thread BLAS worker isolation (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`)**, **pre-warmed persistent actor pools (`clf.warmup()`)**, vectorized `BaseFMNN`/`SingleFMNN`, and strictly train-only MinMax scaling.
Full report: [RAY_PARALLEL_SCALING_REPORT.md](experiments/results/RAY_PARALLEL_SCALING_REPORT.md) | Plot: `figures/ray_multicore_scaling.png`.

### 4.1 RF-FMNN Ray Actor Scaling ($M=32$ Estimators, Warm Actor Pool)

#### Synthetic-Scaled ($N=10000, d=20$)
| Cores ($p$) | Warmup (s) | Train Time (s) | Speedup ($S_p$) | Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.102s | 55.437s | **1.00x** | 100.0% | 47.325s | **93.95%** | 49325 |
| **2** | 3.567s | 31.076s | **1.78x** | 89.2% | 25.963s | **93.95%** | 49325 |
| **4** | 4.280s | 20.003s | **2.77x** | 69.3% | 17.459s | **93.95%** | 49325 |
| **8** | 5.590s | **15.680s** | **3.54x** | 44.2% | **16.484s** | **93.95%** | 49325 |

#### Real High-Dimensional: Digits ($N=1797, d=64$)
| Cores ($p$) | Warmup (s) | Train Time (s) | Speedup ($S_p$) | Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.368s | 3.328s | **1.00x** | 100.0% | 4.350s | **97.78%** | 10059 |
| **2** | 3.751s | 2.057s | **1.62x** | 80.9% | 2.449s | **97.78%** | 10059 |
| **4** | 4.427s | 1.207s | **2.76x** | 69.0% | 1.503s | **97.78%** | 10059 |
| **8** | 5.898s | **0.831s** | **4.00x** | 50.1% | **1.332s** | **97.78%** | 10059 |

#### Small Dataset: BreastCancer ($N=569, d=30$)
| Cores ($p$) | Warmup (s) | Train Time (s) | Speedup ($S_p$) | Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.484s | 0.882s | **1.00x** | 100.0% | 0.248s | **96.49%** | 1828 |
| **2** | 4.405s | 0.488s | **1.81x** | 90.3% | 0.121s | **96.49%** | 1828 |
| **4** | 4.224s | **0.299s** | **2.94x** | 73.6% | **0.086s** | **96.49%** | 1828 |
| **8** | 6.451s | 0.329s | **2.68x** | 33.5% | 0.121s | **96.49%** | 1828 |

**Key Systems Findings**:
- **Eliminating BLAS Thread Oversubscription**: Pinning `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` inside each Ray Actor plus separating one-time actor pool `warmup()` unlocked clean scaling across all 3 datasets: **4.00x on Digits (3.33s to 0.83s)**, **3.54x on Synthetic-Scaled N=10,000 (55.44s to 15.68s)**, and **2.94x on BreastCancer (0.88s to 0.30s at 4 cores)** — approaching the ~5.2x physical compute ceiling of a hybrid 4 P-core + 4 E-core CPU.
- **Zero-Pickling Inference**: Keeping models pinned inside `FMNNWorker` Actor RAM eliminates model serialization between `fit()` and `predict()`, scaling inference to **3.27x speedup on Digits (4.35s to 1.33s)** and **2.87x on N=10,000 (47.33s to 16.48s)**.
- **Digits Hyperparameter Sweep**: Parallel grid search over 8 $\theta$ values on `Digits` reduced sweep time from **50.65s (1 core) to 27.03s (8 cores)** (1.87x speedup; 2.00x at 4 cores).

### 4.2 Estimator Depth Scaling on 8 Cores ($N=10000, d=20$)
| Estimators ($M$) | Warmup (s) | Train Time (s) | Throughput (Bags/sec) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 5.104s | 4.170s | 1.92 | 92.70% | 12276 |
| **16** | 6.413s | 7.686s | 2.08 | 93.45% | 24628 |
| **32** | 5.383s | 13.594s | 2.35 | 93.95% | 49325 |
| **64** | 4.874s | 25.228s | **2.54** | **94.35%** | 98785 |

### 4.3 Hyperbox Consolidation Pruning (`prune=True`, $1.25\theta$ Merge Bound, $M=32$)
Because uncontracted online expansion is monotonic, no two distinct same-class boxes can satisfy union size $\le d \cdot \theta$ at $1.0\times$ scale; relaxing consolidation to $1.25\theta$ merges adjacent same-class fragments while preserving exact sample-weighted centroids:

| Dataset | Unpruned Boxes | Pruned Boxes | Reduction (%) | Unpruned Acc | Pruned Acc | Acc Delta | Unpruned Infer | Pruned Infer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BreastCancer** | 1828 | 1555 | **-14.9%** | 96.49% | 95.61% | -0.88 pp | 0.068s | 0.051s |
| **Digits** | 10059 | 9476 | **-5.8%** | 97.78% | **98.06%** | **+0.28 pp** | 1.004s | 0.949s |
| **Synthetic-Scaled** | 49325 | 49070 | **-0.5%** | 93.95% | 93.95% | +0.00 pp | 10.369s | 13.302s |

---

## 5. Fair Vectorized BaseFMNN Parallelization: Naive Box Pooling vs Sharded Voting

Evaluated on `Synthetic-Scaled (N=10000, d=20)` (8000 train, 2000 test) using **fairly vectorized `BaseFMNN`** (bit-for-bit identical to Simpson 1992, 13.9x faster fit / 22.4x faster predict than naive Python loops):

### 5.1 Naive Data-Partitioned Box Pooling
| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Total Hyperboxes | Box Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8000 | 4.917s | 1.00x | **90.00%** | 2325 | Baseline |
| **2** | 4000 | 5.503s | 0.89x | **90.20%** | 2432 | +107 |
| **4** | 2000 | 4.680s | 1.05x | **90.65%** | 2518 | +193 |
| **8** | 1000 | 6.515s | 0.75x | **90.20%** | 2604 | +279 (+12.0%) |

### 5.2 Fair Sharded Majority-Voting BaseFMNN
| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Acc Delta vs 1-Core | Total Hyperboxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8000 | 4.917s | 1.00x | **90.00%** | Baseline | 2325 |
| **2** | 4000 | 1.633s | 3.01x | **88.85%** | -1.15 pp | 2432 |
| **4** | 2000 | 0.693s | 7.09x | **90.60%** | +0.60 pp | 2518 |
| **8** | 1000 | 0.470s | 10.47x | **90.90%** | +0.90 pp | 2604 |

**Honest Pareto Trade-Off**:
- Once both `BaseFMNN` and `SingleFMNN` are fairly vectorized in NumPy, single-pass `BaseFMNN` trains in **4.92s** (and 8-way sharded-voting `BaseFMNN` in **0.47s** because each worker sees only $1{,}000$ rows), **but accuracy plateaus at 90.00%–90.90%** and naive box-pooling bloats hyperboxes by **+12.0%** (+279 boxes) while slowing down (0.75x).
- **Proposed RF-FMNN** trains $M=32$ full-dataset bootstrap bags across warm Ray Actors in **15.68s** (3.54x faster than serial RF-FMNN's 55.44s), pushing test accuracy to **93.95% (+3.95 pp over 1-core BaseFMNN, +3.05 pp over 8-core Sharded-Voting BaseFMNN)** and **94.35% (+4.35 pp) at $M=64$**.

---

## 6. Honest Technical Critique & Academic Framing

### 6.1 Verified Strengths
1. **Substantial Accuracy Gain on Large Overlapping Tabular Data**: On $N=10{,}000$ (`Synthetic-Scaled`), RF-FMNN outperforms vectorized 1-core BaseFMNN by **+3.95 pp to +4.35 pp** (93.95%–94.35% vs 90.00%) and 8-core Sharded-Voting BaseFMNN by **+3.05 pp** (93.95% vs 90.90%).
2. **Statistically Significant High-Dimensional Performance**: On 64-dimensional `Digits`, RF-FMNN (Bagging) achieves **98.39%** accuracy, beating Random Forest (97.44%) with $p = 0.0105$. On 30-dimensional `BreastCancer`, RF-FMNN leads all models at **96.84%**.
3. **Clean Multi-Core Ray Actor Scaling**: Under single-thread BLAS isolation and warm actor pools, RF-FMNN achieves **4.00x speedup on Digits** (3.33s to 0.83s) and **3.54x speedup on N=10,000** (55.44s to 15.68s) with zero accuracy degradation.
4. **Fair Engineering & Zero-Leakage Rigor**: Both `BaseFMNN` and `RF-FMNN` are vectorized in NumPy, normalization is strictly in-fold, centroids use $O(d)$ running sums, and all reports are dynamically generated from experimental JSON.

### 6.2 Honest Limitations
1. **No Statistically Significant Accuracy Win Over Basic FMNN on Small Benchmarks**: Across all 6 CV datasets, paired t-tests against Basic FMNN yield $p > 0.05$ (`Digits` +0.28pp $p=0.27$, `BreastCancer` +1.05pp $p=0.28$, `Wine` -2.25pp $p=0.10$).
2. **Hyperbox Memory Footprint**: Storing $M=30\text{--}32$ bags generates $20\times\text{--}23\times$ more hyperboxes than single BaseFMNN (e.g., 9441.6 vs 408.8 on Digits). `max_features='sqrt'` cuts box count by ~40% on high-$d$ data (but hurts 2D manifolds), and $1.25\theta$ consolidation pruning trims 5.8%–14.9% on real datasets.
3. **Ray Actor Warmup Cost**: Spawning 8 cold Ray Actor OS processes takes ~4.9–6.5s. Ray Actors pay off cleanly for persistent services, large $N \ge 5{,}000$, or large ensemble sweeps ($M \ge 32$).

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
