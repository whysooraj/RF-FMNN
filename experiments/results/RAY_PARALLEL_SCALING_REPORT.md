# Ray Parallel Processing Benchmark: 1, 2, 4, and 8 Cores Scaling Analysis

Evaluated on a 12-core Linux host comparing Standard Sequential FMNN (Simpson 1992), Naive Box-Pooling BaseFMNN, Fair Sharded Majority-Voting BaseFMNN, and Ray-Distributed Proposed RF-FMNN (with strictly train-only MinMax scaling).

## 1. Executive Summary

- **Proposed RF-FMNN Parallel Scaling**: On `Synthetic-Scaled (N=5000, d=20)` ($M=32$ estimators), Proposed RF-FMNN reduced wall-clock training time from 14.549s (1 core) to 6.709s (8 cores), achieving **2.17x speedup** (27.1% parallel efficiency) with consistent test accuracy (94.90% at 1 core vs 94.90% at 8 cores). Across all 3 evaluated datasets, 8-core speedup ranged from 0.77x to 2.17x.
- **Normal BaseFMNN Naive Box Pooling**: Sharding `Synthetic-Scaled (N=5000, d=20)` across 8 cores and pooling uncontracted hyperboxes achieved 7.05x speedup (28.034s to 3.978s), shifted test accuracy by -1.70 pp (92.50% to 90.80%), and expanded total hyperboxes from 1235 to 1367 (+132 boxes, 1.11x).
- **Normal BaseFMNN Fair Sharded Majority Voting**: Training 8 independent BaseFMNN models on disjoint $N/p$ shards (500 training samples/shard vs 4000 full-dataset samples) and combining via majority vote achieved 26.84x speedup with test accuracy 91.80% (-0.70 pp vs 1-core BaseFMNN). Compared against Sharded-Voting BaseFMNN at 8 cores, Proposed RF-FMNN (using full $N$-size bootstrap bags) achieved 94.90% accuracy (+3.10 pp difference).
- **Parallel Hyperparameter Sweep**: Evaluating 8 $\theta$ configurations in parallel reduced wall-clock sweep time from 4.808s (1 core) to 4.255s (8 cores), yielding **1.13x speedup** (14.1% efficiency).
- **Estimator Throughput Scaling (8 Cores)**: Scaling from $M=8$ to $M=64$ estimators on `Synthetic-Scaled (N=5000, d=20)` reached **5.52 bags/sec** (11.592s wall time, 95.00% test accuracy).

## 2. Proposed RF-FMNN Core Scaling: Wall-Clock Time, Speedup, and Efficiency

### Dataset: BreastCancer (N=569, d=30) ($M=32$ Estimators)

| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.472s | **1.00x** | 100.0% | 0.188s | 96.49% | 1828 |
| **2** | 2.479s | **1.00x** | 49.9% | 0.118s | 96.49% | 1828 |
| **4** | 2.290s | **1.08x** | 27.0% | 0.084s | 96.49% | 1828 |
| **8** | 3.212s | **0.77x** | 9.6% | 0.067s | 96.49% | 1828 |

### Dataset: Digits (N=1797, d=64) ($M=32$ Estimators)

| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.095s | **1.00x** | 100.0% | 3.197s | 97.78% | 10059 |
| **2** | 3.229s | **1.27x** | 63.4% | 2.339s | 97.78% | 10059 |
| **4** | 2.903s | **1.41x** | 35.3% | 1.362s | 97.78% | 10059 |
| **8** | 3.513s | **1.17x** | 14.6% | 1.104s | 97.78% | 10059 |

### Dataset: Synthetic-Scaled (N=5000, d=20) ($M=32$ Estimators)

| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.549s | **1.00x** | 100.0% | 9.281s | 94.90% | 25597 |
| **2** | 10.290s | **1.41x** | 70.7% | 5.751s | 94.90% | 25597 |
| **4** | 6.766s | **2.15x** | 53.8% | 3.462s | 94.90% | 25597 |
| **8** | 6.709s | **2.17x** | 27.1% | 2.923s | 94.90% | 25597 |

## 3. Parallelizing Normal BaseFMNN on Synthetic-Scaled (N=5000, d=20): Naive Box Pooling vs Fair Sharded Voting

### 3A. Naive Data-Partitioned Box Pooling (Cross-Shard Contraction Failure)

| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Total Hyperboxes | Box Delta vs 1-Core |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4000 | 28.034s | 1.00x | **92.50%** | 1235 | Baseline |
| **2** | 2000 | 9.982s | 2.81x | **90.70%** | 1269 | +34 |
| **4** | 1000 | 4.519s | 6.20x | **91.70%** | 1316 | +81 |
| **8** | 500 | 3.978s | 7.05x | **90.80%** | 1367 | +132 |

### 3B. Fair Sharded Majority-Voting BaseFMNN (Reduced Per-Learner Coverage $N/p$)

| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Acc Delta vs 1-Core (pp) | Total Hyperboxes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4000 | 28.034s | 1.00x | **92.50%** | Baseline | 1235 |
| **2** | 2000 | 8.368s | 3.35x | **89.90%** | -2.60 | 1269 |
| **4** | 1000 | 2.677s | 10.47x | **90.70%** | -1.80 | 1316 |
| **8** | 500 | 1.044s | 26.84x | **91.80%** | -0.70 | 1367 |

## 4. Parallel Hyperparameter Tuning (8 $\theta$ Configurations)

| Cores ($p$) | Grid Search Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) |
| :---: | :---: | :---: | :---: |
| **1** | 4.808s | **1.00x** | 100.0% |
| **2** | 3.907s | **1.23x** | 61.5% |
| **4** | 3.409s | **1.41x** | 35.3% |
| **8** | 4.255s | **1.13x** | 14.1% |

## 5. Estimator Scaling on 8 Cores (Synthetic-Scaled (N=5000, d=20) Throughput Analysis)

| Estimators ($M$) | Train Time (s) | Throughput (Bags/sec) | Test Accuracy | Total Boxes |
| :---: | :---: | :---: | :---: | :---: |
| **8** | 3.856s | **2.07** | 93.70% | 6372 |
| **16** | 5.885s | **2.72** | 95.50% | 12752 |
| **32** | 7.578s | **4.22** | 94.90% | 25597 |
| **64** | 11.592s | **5.52** | 95.00% | 51322 |

