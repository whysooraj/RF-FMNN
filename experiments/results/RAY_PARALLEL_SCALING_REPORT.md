# Ray Parallel Processing Benchmark: 1, 2, 4, and 8 Cores Scaling Analysis

Evaluated on a 12-core Linux host comparing Vectorized Sequential FMNN (Simpson 1992), Naive Box-Pooling BaseFMNN, Fair Sharded Majority-Voting BaseFMNN, and Ray-Distributed Proposed RF-FMNN (with strictly train-only MinMax scaling, single-thread BLAS worker isolation, and pre-warmed persistent Ray actor pools).

## 1. Executive Summary

- **Proposed RF-FMNN Parallel Scaling**: On `Synthetic-Scaled (N=10000, d=20)` ($M=32$ estimators), Proposed RF-FMNN reduced steady-state wall-clock training time from 55.437s (1 core) to 15.680s (8 cores), achieving **3.54x speedup** (44.2% parallel efficiency) with consistent test accuracy (93.95% at 1 core vs 93.95% at 8 cores). Across all 3 evaluated datasets, 8-core speedup ranged from 2.68x to 4.00x.
- **Warm Actor Pool Measurement Protocol**: Persistent Ray worker actors are pre-warmed via `clf.warmup()` prior to `fit()` timing so steady-state parallel compute scaling is measured without cold OS worker spawn overhead; separate `warmup_time_s` is reported in all scaling tables (3.102s at 1 core, 5.590s at 8 cores on `Synthetic-Scaled (N=10000, d=20)`).
- **Normal BaseFMNN Naive Box Pooling**: Sharding `Synthetic-Scaled (N=10000, d=20)` across 8 cores and pooling uncontracted hyperboxes achieved 0.75x speedup (4.917s to 6.515s), shifted test accuracy by +0.20 pp (90.00% to 90.20%), and expanded total hyperboxes from 2325 to 2604 (+279 boxes, 1.12x).
- **Normal BaseFMNN Fair Sharded Majority Voting**: Training 8 independent BaseFMNN models on disjoint $N/p$ shards (1000 training samples/shard vs 8000 full-dataset samples) and combining via majority vote achieved 10.47x speedup with test accuracy 90.90% (+0.90 pp vs 1-core BaseFMNN). Compared against Sharded-Voting BaseFMNN at 8 cores, Proposed RF-FMNN (using full $N$-size bootstrap bags) achieved 93.95% accuracy (+3.05 pp difference).
- **Parallel Hyperparameter Sweep (`Digits (N=1797, d=64)`)**: Evaluating 8 $\theta$ configurations in parallel reduced wall-clock sweep time from 50.651s (1 core) to 27.030s (8 cores), yielding **1.87x speedup** (23.4% efficiency).
- **Estimator Throughput Scaling (8 Cores)**: Scaling from $M=8$ to $M=64$ estimators on `Synthetic-Scaled (N=10000, d=20)` reached **2.54 bags/sec** (25.228s wall time, 94.35% test accuracy).
- **Hyperbox Consolidation Pruning ($M=32$, 8 Cores)**: Enabling `prune=True` reduced total ensemble hyperboxes by **0.5% to 14.9%** across all 3 datasets with test accuracy deltas between -0.88 pp and +0.28 pp.

## 2. Proposed RF-FMNN Core Scaling: Wall-Clock Time, Speedup, and Efficiency

Each run pre-warms its persistent Ray actor pool (`clf.warmup()`) under single-thread BLAS worker isolation before timing `clf.fit()`, separating one-time actor startup (`Warmup Time`) from steady-state parallel training (`Train Time`).

### Dataset: BreastCancer (N=569, d=30) ($M=32$ Estimators)

| Cores ($p$) | Warmup Time (s) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.484s | 0.882s | **1.00x** | 100.0% | 0.248s | 96.49% | 1828 |
| **2** | 4.405s | 0.488s | **1.81x** | 90.3% | 0.121s | 96.49% | 1828 |
| **4** | 4.224s | 0.299s | **2.94x** | 73.6% | 0.086s | 96.49% | 1828 |
| **8** | 6.451s | 0.329s | **2.68x** | 33.5% | 0.121s | 96.49% | 1828 |

### Dataset: Digits (N=1797, d=64) ($M=32$ Estimators)

| Cores ($p$) | Warmup Time (s) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.368s | 3.328s | **1.00x** | 100.0% | 4.350s | 97.78% | 10059 |
| **2** | 3.751s | 2.057s | **1.62x** | 80.9% | 2.449s | 97.78% | 10059 |
| **4** | 4.427s | 1.207s | **2.76x** | 69.0% | 1.503s | 97.78% | 10059 |
| **8** | 5.898s | 0.831s | **4.00x** | 50.1% | 1.332s | 97.78% | 10059 |

### Dataset: Synthetic-Scaled (N=10000, d=20) ($M=32$ Estimators)

| Cores ($p$) | Warmup Time (s) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.102s | 55.437s | **1.00x** | 100.0% | 47.325s | 93.95% | 49325 |
| **2** | 3.567s | 31.076s | **1.78x** | 89.2% | 25.963s | 93.95% | 49325 |
| **4** | 4.280s | 20.003s | **2.77x** | 69.3% | 17.459s | 93.95% | 49325 |
| **8** | 5.590s | 15.680s | **3.54x** | 44.2% | 16.484s | 93.95% | 49325 |

## 3. Parallelizing Normal BaseFMNN on Synthetic-Scaled (N=10000, d=20): Naive Box Pooling vs Fair Sharded Voting

### 3A. Naive Data-Partitioned Box Pooling (Cross-Shard Contraction Failure)

| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Total Hyperboxes | Box Delta vs 1-Core |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8000 | 4.917s | 1.00x | **90.00%** | 2325 | Baseline |
| **2** | 4000 | 5.503s | 0.89x | **90.20%** | 2432 | +107 |
| **4** | 2000 | 4.680s | 1.05x | **90.65%** | 2518 | +193 |
| **8** | 1000 | 6.515s | 0.75x | **90.20%** | 2604 | +279 |

### 3B. Fair Sharded Majority-Voting BaseFMNN (Reduced Per-Learner Coverage $N/p$)

| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Acc Delta vs 1-Core (pp) | Total Hyperboxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8000 | 4.917s | 1.00x | **90.00%** | Baseline | 2325 |
| **2** | 4000 | 1.633s | 3.01x | **88.85%** | -1.15 | 2432 |
| **4** | 2000 | 0.693s | 7.09x | **90.60%** | +0.60 | 2518 |
| **8** | 1000 | 0.470s | 10.47x | **90.90%** | +0.90 | 2604 |

## 4. Parallel Hyperparameter Tuning on Digits (N=1797, d=64) (8 $\theta$ Configurations)

| Cores ($p$) | Grid Search Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) |
| :---: | :---: | :---: | :---: |
| **1** | 50.651s | **1.00x** | 100.0% |
| **2** | 31.059s | **1.63x** | 81.5% |
| **4** | 25.278s | **2.00x** | 50.1% |
| **8** | 27.030s | **1.87x** | 23.4% |

## 5. Estimator Scaling on 8 Cores (Synthetic-Scaled (N=10000, d=20) Throughput Analysis)

| Estimators ($M$) | Warmup Time (s) | Train Time (s) | Throughput (Bags/sec) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 5.104s | 4.170s | **1.92** | 92.70% | 12276 |
| **16** | 6.413s | 7.686s | **2.08** | 93.45% | 24628 |
| **32** | 5.383s | 13.594s | **2.35** | 93.95% | 49325 |
| **64** | 4.874s | 25.228s | **2.54** | 94.35% | 98785 |

## 6. Hyperbox Consolidation Pruning Evaluation ($M=32$, 8 Cores)

| Dataset | Unpruned Boxes | Pruned Boxes | Box Reduction (%) | Unpruned Acc | Pruned Acc | Acc Delta (pp) | Unpruned Infer (s) | Pruned Infer (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BreastCancer (N=569, d=30) | 1828 | 1555 | **14.9%** | 96.49% | 95.61% | -0.88 | 0.0683s | 0.0513s |
| Digits (N=1797, d=64) | 10059 | 9476 | **5.8%** | 97.78% | 98.06% | +0.28 | 1.0042s | 0.9492s |
| Synthetic-Scaled (N=10000, d=20) | 49325 | 49070 | **0.5%** | 93.95% | 93.95% | +0.00 | 10.3690s | 13.3024s |

