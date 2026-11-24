# Comprehensive Analysis: Classic FMNN (Simpson 1992) vs. Proposed RF-FMNN

This document presents the theoretical, algorithmic, and computational differences between the classic **Fuzzy Min-Max Neural Network (Simpson 1992)** and the **Proposed RF-FMNN Ensemble**.

---

## 1. Architectural Comparison Matrix

| Property | Normal Base FMNN (Simpson 1992) | Proposed RF-FMNN |
| :--- | :--- | :--- |
| **Model Structure** | Single monolithic neural network | Ensemble of $M$ independent base FMNN learners |
| **Data Sampling** | 100% training data fed in single sequential stream | Bootstrap sampling ($N$ draws with replacement per bag) until full training coverage (Coupon-Collector stopping rule: expected bags $\mathbb{E}[B] \approx \ln(N) + \gamma$, where $\gamma \approx 0.5772$) |
| **Feature Modes** | Always full $d$ features | Two modes: row-bagging mode (`max_features=None`, full $d$ features) and feature-subspace mode (`max_features='sqrt'`, $k = \lfloor\sqrt{d}\rfloor$ features per learner) |
| **Normalization Discipline** | Requires $[0, 1]^d$ inputs | `MinMaxScaler` fitted strictly on the training fold inside cross-validation; test observations clipped to $[0, 1]$ to prevent data leakage |
| **Hyperbox Contraction** | **Active (4-case geometric shrinkage)** | **Completely Removed (Uncontracted)** |
| **Overlap Resolution** | Destructive coordinate clipping ($v_j$ or $w_j$ modified) | Preserved overlaps; resolved via nearest arithmetic data centroid |
| **Centroid Definition** | Geometric bounding-box midpoint: $(V + W) / 2$ | Arithmetic mean of assigned training observations: $\frac{1}{\|P_j\|} \sum_{x \in P_j} x$, tracked incrementally via `UncontractedHyperbox` (`sample_sum / sample_count`) in $O(d)$ memory with exact bootstrap multiplicity |
| **Input Order Sensitivity** | **High** (different permutations produce completely different boxes) | **Low** (bootstrap resampling and ensemble voting average out permutation bias) |
| **Parallel Scalability** | **Inherently sequential** across samples due to stateful overlap/contraction coupling | **Embarrassingly parallel** via Stateful Ray Actors (`@ray.remote class FMNNWorker`) + zero-copy Plasma Object Store (`ray.put`), pinning models in worker RAM for zero pickling between `fit()` and `predict()` |
| **Distributed Data Sharding** | **Degrades under both strategies**: naive box-pooling causes unresolved inter-class overlaps and box explosion; sharded-voting shrinks per-worker training data to $N/P$ as core count $P$ grows | **Native**: Independent bootstrap bags train across persistent Ray actors with zero inter-worker communication and core-count-invariant accuracy |
| **Inference Voting** | Single model membership / winner-take-all | Plurality / majority voting across $M$ estimators with probability confidence |

---

## 2. Algorithmic Differences: Step-by-Step

```
Normal FMNN (Simpson 1992)
---------------------------
Input Stream x_t ---> [ Expansion Test (theta) ]
                             |
                   Expansion allowed?
                   /                \
                Yes                  No
                 |                    |
        Expand Box (V, W)       Create New Box
                 |
     [ Overlap Test with Other Classes ]
                 |
        Overlap detected?
         /              \
       Yes               No
        |                 |
  Apply 1 of 4 Contractions  Done
  (Shrinks box dimensions)

Proposed RF-FMNN (with Ray Actor Execution)
-------------------------------------------
Train Fold D ---> [ Fit MinMaxScaler on Train Fold Only ]
                                   |
                  [ ray.put(X_train, y_train) -> Plasma Store ]
                                   |
                  [ Bootstrap Bags until 100% Coverage ]
                  (E[B] ~ ln(N) + gamma bags; N draws/bag;
                   max_features=None (d) or 'sqrt' (floor(sqrt(d))))
                                   |
                   +---------------+---------------+
                   |               |               |
            FMNNWorker 1    FMNNWorker 2    FMNNWorker P   <-- Stateful Ray Actors
                   |               |               |
             Expand Boxes    Expand Boxes    Expand Boxes
             (NO CONTRACTION)(NO CONTRACTION)(NO CONTRACTION)
                   |               |               |
             Update Running  Update Running  Update Running
             sample_sum /    sample_sum /    sample_sum /
             sample_count    sample_count    sample_count
                   |               |               |
             [ Pin Trained Models in Worker RAM (Zero Pickling) ]
                   \               |               /
                    +--------------+--------------+
                                   |
             Test Fold -> Transform & Clip to [0, 1] -> ray.put(X_test)
                                   |
                        Local Worker Inference
                     -> Gather Compact Vote Arrays
                        -> Ensemble Plurality Vote
```

---

## 3. Mathematical Differences in Overlap Handling

### Classic Simpson (1992) Contraction Heuristic
When an expanded box $B_j$ (class $c_1$) overlaps with box $B_k$ (class $c_2$), Simpson checks 4 geometric cases along the dimension of minimum overlap $\Delta^{\min}$:

$$\Delta^{\min} = \min_{i} \left( \min(w_{ji} - v_{ki}, w_{ki} - v_{ji}) \right)$$

Depending on which of 4 cases is satisfied, one box coordinate is truncated:
- Case 1: $v_{ji} < v_{ki} < w_{ji} < w_{ki} \implies w_{ji}^{\text{new}} = v_{ki}^{\text{new}} = \frac{w_{ji} + v_{ki}}{2}$
- Case 2: $v_{ki} < v_{ji} < w_{ki} < w_{ji} \implies w_{ki}^{\text{new}} = v_{ji}^{\text{new}} = \frac{w_{ki} + v_{ji}}{2}$
- Case 3: $v_{ji} \le v_{ki} \le w_{ki} \le w_{ji} \implies$ shrink $v_{ji}$ or $w_{ji}$
- Case 4: $v_{ki} \le v_{ji} \le w_{ji} \le w_{ki} \implies$ shrink $v_{ki}$ or $w_{ki}$

**Consequence**: Contraction creates artificial boundaries, clips regions containing genuine data points, and produces severe sensitivity to the exact order in which observations are presented.

### Proposed RF-FMNN Centroid Resolution
Hyperboxes are never shrunk. Overlaps between classes are permitted. Letting $P_j$ denote the multiset of training observations assigned to hyperbox $B_j$ (including bootstrap repetitions), its centroid is the exact multiplicity-weighted arithmetic mean:

$$C_j = \frac{1}{|P_j|} \sum_{x \in P_j} x = \frac{\text{sample\_sum}_j}{\text{sample\_count}_j}$$

In implementation, `UncontractedHyperbox` maintains only the running arithmetic sum vector (`sample_sum`) and integer counter (`sample_count`). This preserves exact bootstrap multiplicity while consuming only $O(d)$ memory per hyperbox (or $O(k)$ in feature-subspace mode) without retaining raw training copies.

If query $x$ lies in an overlap between multiple boxes $\{B_j\}_{j \in \mathcal{O}}$ of different classes:

$$c^*(x) = \arg\min_{B_j \in \mathcal{O}} \|x - C_j\|_2$$

**Consequence**: Clean, data-density-driven boundary resolution without geometric distortions, raw sample storage overhead, or contraction-induced information loss.

---

## 4. Multi-Core Scaling & Ray Parallelism

### Proposed RF-FMNN with Stateful Ray Actors (`FMNNWorker`)
RF-FMNN is natively suited to multi-core execution because every bootstrap bag trains an independent `SingleFMNN` learner (in either row-bagging mode `max_features=None` or feature-subspace mode `max_features='sqrt'`) with zero cross-learner synchronization during training:

1. **Zero-Copy Shared Data (`ray.put`)**: The normalized training arrays $(X_{\text{train}}, y_{\text{train}})$ are placed once into Ray's Plasma Object Store so all workers read them from shared memory without duplicating payloads.
2. **Stateful Ray Actors (`@ray.remote class FMNNWorker`)**: A fixed pool of persistent worker actors receives lightweight bootstrap index chunks (and feature subspace indices when `max_features='sqrt'`). Each actor trains its assigned learners locally and **pins the trained hyperbox models inside worker RAM**.
3. **Zero-Pickling Inference**: During `predict()` and `predict_proba()`, only the Plasma object reference to `X_test` is dispatched to the actors. Each actor evaluates its pinned models locally and returns only compact prediction vote arrays, eliminating model serialization/pickling overhead between `fit()` and `predict()`.
4. **Core-Count-Invariant Accuracy**: Because the ensemble of bootstrap bags (expected count $\mathbb{E}[B] \approx \ln(N) + \gamma$ under Coupon-Collector coverage) is mathematically identical regardless of how bags are partitioned across $P$ workers, classification accuracy remains unchanged across core counts.

### Why Normal BaseFMNN (Simpson 1992) Fails Under Parallel Data Partitioning
Standard FMNN is inherently sequential because each incoming observation can trigger global overlap checks and contractions against previously constructed boxes of all other classes. Attempting to parallelize `BaseFMNN` by partitioning the training set across $P$ workers exposes two distinct failure modes:

1. **Naive Box-Pooling Sharded BaseFMNN**:
   - Training rows are partitioned across $P$ workers, each builds local contracted hyperboxes, and all resulting boxes are pooled into a single classifier.
   - Because workers lack cross-shard visibility, hyperboxes created on separate shards never undergo Simpson overlap testing or contraction against one another.
   - Pooling leaves unresolved inter-class overlaps across shards and creates redundant fragmented hyperboxes (box explosion), degrading predictive accuracy.
2. **Sharded-Voting BaseFMNN**:
   - Training rows are partitioned into $P$ disjoint shards, one `BaseFMNN` is trained per shard, and predictions are combined by majority voting.
   - While this avoids cross-shard box corruption, each learner only sees $N / P$ observations. As core count $P$ grows, every learner trains on a progressively smaller fragment of the data distribution, increasing per-learner bias and degrading ensemble accuracy. By contrast, RF-FMNN gives every learner a full $N$-sample bootstrap draw with complete dataset coverage guaranteed across the ensemble.

### Empirical Multi-Core Scaling Results
For up-to-date wall-clock training/inference timings, speedup curves, parallel efficiency ($E_p$), hyperbox counts, and accuracy comparisons across core counts (covering Ray Actor RF-FMNN alongside naive box-pooling and sharded-voting BaseFMNN baselines), refer to [`experiments/results/RAY_PARALLEL_SCALING_REPORT.md`](experiments/results/RAY_PARALLEL_SCALING_REPORT.md) and [`experiments/results/ray_scaling_results.json`](experiments/results/ray_scaling_results.json).

---

## 5. Visualizations

Generated publication figures available in `figures/`:
1. `figures/ray_multicore_scaling.png`:
   - Multi-core scaling comparison showing speedup curves, wall-clock latency reduction, parallel compute efficiency ($E_p$), and BaseFMNN sharding degradation.
2. `figures/normal_vs_rf_fmnn_concept.png`:
   - Side-by-side conceptual diagram contrasting Simpson 1992 geometric clipping against RF-FMNN centroid-based overlap resolution.
