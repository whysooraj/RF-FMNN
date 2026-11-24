# Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN): Methodology, Parallel Architecture, and Empirical Evaluation

**Authors**: RF-FMNN Research Contributors  
**Repository**: `https://github.com/whysooraj/RF-FMNN`  
**Classification**: Neuro-Fuzzy Systems, Granular Computing, Ensemble Learning, Distributed Computing  

---

## Abstract

Fuzzy Min-Max Neural Networks (FMNN), introduced by Simpson (1992), combine fuzzy set theory with online neural pattern classification through interpretable hyperbox decision regions. However, classic FMNN relies on a sequential overlap-contraction heuristic that truncates colliding hyperbox boundaries across opposing classes, inducing presentation-order sensitivity, geometric fragmentation in high-dimensional spaces, and an inherent sequential bottleneck that resists multi-core parallelization. In this paper, we formulate and empirically evaluate the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN). Proposed RF-FMNN eliminates overlap contraction entirely by training independent uncontracted FMNN learners over bootstrap resamples (under either a fixed estimator budget $M$ or a Coupon-Collector full-coverage stopping criterion) with optional random feature subspace selection ($k = \max(1, \lfloor\sqrt{d}\rfloor)$). Ambiguous multi-class hyperbox overlaps are resolved via nearest Euclidean distance to $O(d)$ running-sum arithmetic centroids that preserve exact bootstrap multiplicity without retaining raw training vectors. For multi-core execution, we pair vectorized NumPy training and batch inference with stateful Ray Actors (`FMNNWorker`), zero-copy Plasma Object Store sharing, and single-threaded BLAS worker isolation (`OMP_NUM_THREADS=1`). Under a strictly leak-free 5-fold stratified cross-validation protocol (`MinMaxScaler` fit inside each training fold, test features clipped to $[0, 1]$) across six benchmark datasets (Iris, Wine, Breast Cancer Wisconsin, Handwritten Digits, Two Moons, and Concentric Circles) and five models, Proposed RF-FMNN (Bagging, $M=30$) achieves the highest accuracy on 64-dimensional Handwritten Digits (**98.39% +/- 0.48%**, outperforming Random Forest at 97.44% with paired Student $t$-test $p = 0.0105$), leads all models on 30-dimensional Breast Cancer Wisconsin (**96.84% +/- 1.31%**), and maintains competitive accuracy across low-dimensional manifolds. We further report honest statistical significance findings, quantify the memory-versus-accuracy trade-off of `max_features='sqrt'`, and examine structural limitations of parallelizing classic contracting FMNN via data sharding.

---

## 1. Introduction and Background

Fuzzy Min-Max Neural Networks (Simpson, 1992) are a foundational family of granular neuro-fuzzy classifiers in which each pattern class is represented as a union of $d$-dimensional axis-aligned hyperboxes. Each hyperbox $B_j$ is parameterized by its minimum and maximum coordinate extrema:

$$V_j = (v_{j1}, v_{j2}, \dots, v_{jd}), \quad W_j = (w_{j1}, w_{j2}, \dots, w_{jd}) \in [0, 1]^d$$

The fuzzy membership degree of an input pattern $x = (x_1, \dots, x_d) \in [0, 1]^d$ to hyperbox $B_j$ is defined by Simpson's ramp membership function:

$$b_j(x) = \frac{1}{2d} \sum_{i=1}^d \left[ \max\left(0, 1 - \max\left(0, \gamma \min(1, x_i - w_{ji})\right)\right) + \max\left(0, 1 - \max\left(0, \gamma \min(1, v_{ji} - x_i)\right)\right) \right]$$

where $\gamma > 0$ is the sensitivity parameter governing how rapidly membership decays outside the hyperbox boundary. When $x$ lies inside $B_j$ (i.e., $v_{ji} \le x_i \le w_{ji}$ for all $i$), $b_j(x) = 1$.

Training a classic Basic FMNN (Simpson, 1992) proceeds sequentially over training vectors through three coupled stages:
1. **Expansion**: Identify the same-class hyperbox with maximum membership and expand it to absorb $x$ if the $L_1$ perimeter growth remains bounded by $d \cdot \theta$, where $\theta \in (0, 1]$ is the maximum hyperbox size parameter; otherwise initialize a point hyperbox $V = W = x$.
2. **Overlap Testing**: Scan all opposing-class hyperboxes across all $d$ dimensions to detect geometric intersection.
3. **Contraction**: When overlap is detected, apply Simpson's four-case contraction rule along the dimension of minimal overlap to force disjointness between opposing classes.

Although compact for low-dimensional streams, this sequential contraction mechanism creates three structural liabilities:
- **Geometric Distortion**: Truncating boundaries along a single axis can excise high-density interior regions and fracture coherent class clusters in higher dimensions.
- **Input-Order Dependence**: Pairwise collision history depends strictly on sample arrival order, causing variance across permutations.
- **Parallelization Barrier**: Global overlap testing and contraction couple every sample update to the full hyperbox set. Partitioning training data across worker processes without altering the algorithm either omits cross-shard contractions (causing uncontracted collisions and hyperbox bloat) or shrinks each learner's sample coverage to $N/p$ (degrading generalization).

To address these limitations rigorously, we evaluate **Proposed RF-FMNN** under a leak-free 5-fold stratified cross-validation protocol across six benchmarks spanning $d=2$ to $d=64$ dimensions: Iris ($N=150, d=4, C=3$), Wine ($N=178, d=13, C=3$), Breast Cancer Wisconsin ($N=569, d=30, C=2$), Handwritten Digits ($N=1797, d=64, C=10$), Two Moons ($N=300, d=2, C=2$), and Concentric Circles ($N=300, d=2, C=2$). Five models are evaluated under identical fold splits:
1. **Basic FMNN (Simpson 1992)**
2. **Decision Tree (CART)**
3. **Random Forest (Breiman, $M=30$)**
4. **Proposed RF-FMNN (Bagging, $M=30$)**
5. **Proposed RF-FMNN (Subspace $\lfloor\sqrt{d}\rfloor$, $M=30$)**

---

## 2. Proposed RF-FMNN Methodology

Proposed RF-FMNN replaces global sequential contraction with bootstrap ensemble aggregation, uncontracted hyperbox learning, and centroid-based overlap disambiguation.

```
Raw Training Fold (N_tr x d)
     |
     v
In-Fold MinMaxScaler Fit -> Scale X_train to [0, 1]^d (Clip X_test to [0, 1]^d)
     |
     v
Bootstrap Bag Specification (Fixed M or Coupon-Collector Full Coverage)
+ Optional Feature Subspace Selection (k = max(1, floor(sqrt(d))))
     |
     v
Independent Uncontracted SingleFMNN Learners (Vectorized NumPy fit())
- Expansion bounded by d * theta; NO overlap contraction
- O(d) running-sum centroid tracking (sample_sum / sample_count)
- Optional theta-preserving same-class hyperbox consolidation pruning
     |
     v
Inference per Learner (Vectorized Batch predict())
- Single-class containment -> direct class assignment
- Multi-class overlap -> nearest Euclidean centroid argmin ||x - C_j||_2
- Outside all boxes -> Simpson fuzzy membership fallback (centroid tie-break)
     |
     v
Ensemble Majority / Plurality Voting (Deterministic Lowest-Label Tie-Break)
     |
     v
Final Class Prediction
```

### 2.1. Leak-Free Feature Scaling Protocol

Because FMNN expansion bounds depend directly on unit-hypercube coordinates $[0, 1]^d$, scaling must be performed strictly inside each cross-validation fold. For every fold $f$, a `MinMaxScaler` is fit exclusively on the training partition $X_{\text{train}}^{(f)}$ and applied to $X_{\text{test}}^{(f)}$. Out-of-range test coordinates are clipped to $[0, 1]$:

$$\tilde{x}_{\text{test}} = \text{clip}\!\left(\frac{x_{\text{test}} - \min(X_{\text{train}}^{(f)})}{\max(X_{\text{train}}^{(f)}) - \min(X_{\text{train}}^{(f)})},\; 0,\; 1\right)$$

This prevents distributional leakage of test extrema into training hyperbox bounds.

### 2.2. Bootstrap Sampling and Coupon-Collector Coverage Analysis

Given a training fold of $N$ observations, each bootstrap bag $b$ draws $N$ indices uniformly with replacement. Asymptotically, the out-of-bag exclusion probability per bag is:

$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679$$

so each bag contains approximately $63.2\%$ unique training observations. Proposed RF-FMNN supports two bag-budget regimes:

1. **Coupon-Collector Full-Coverage Rule (`n_estimators=None`)**: Bags are generated iteratively until the union of sampled index sets covers every training observation ID (capped by a safety limit):
   $$\bigcup_{b=1}^{B} I_b = \{1, 2, \dots, N\}$$
   By the classical Coupon-Collector problem, covering all $N$ distinct IDs requires approximately $N(\ln N + \gamma)$ **individual sample draws** in expectation, where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant (distinct from the FMNN membership sensitivity parameter). Because each bootstrap bag performs $N$ draws simultaneously, the **expected number of bags** $B$ divides the total draw count by $N$:
   $$\mathbb{E}[B] \approx \ln(N) + \gamma, \quad \gamma \approx 0.5772$$
   For $N \in [120, 1500]$, this yields $\mathbb{E}[B] \approx 5.4$ to $7.9$ bags, which is explicitly distinct from the $N(\ln N + \gamma)$ total individual sample draws.
2. **Matched Ensemble Budget (`n_estimators=M`)**: For controlled comparative evaluation against Breiman's Random Forest and lower-variance voting, we set $M = 30$ independent bootstrap learners (`n_estimators=30`).

### 2.3. Optional Feature Subspace Sampling (`max_features='sqrt'`)

To evaluate variance decorrelation and hyperbox memory reduction in higher dimensions, Proposed RF-FMNN supports per-learner random feature subspaces without replacement of size:

$$k = \max\left(1, \lfloor\sqrt{d}\rfloor\right)$$

For a fixed ensemble of $M$ learners, the probability that any specific feature $i \in \{1, \dots, d\}$ is never selected across the entire ensemble is:

$$P(\text{feature } i \text{ excluded from all } M \text{ learners}) = \left(1 - \frac{k}{d}\right)^M$$

For $d = 30, k = 5, M = 30$, this exclusion probability is $(25/30)^{30} \approx 0.0042$ ($0.42\%$). For $d = 64, k = 8, M = 30$, it is $(56/64)^{30} \approx 0.0182$ ($1.82\%$). Crucially, on two-dimensional manifolds ($d = 2$), $k = \lfloor\sqrt{2}\rfloor = 1$, forcing every learner to project onto a single coordinate axis.

### 2.4. Uncontracted Expansion and $O(d)$ Running-Sum Centroids

Each learner (`SingleFMNN`) processes its assigned bootstrap bag sequentially without overlap testing or contraction. For an incoming sample $(x, c)$, let $\mathcal{B}_c$ denote the existing hyperboxes of class $c$. A candidate box $B_j = (V_j, W_j) \in \mathcal{B}_c$ is eligible for expansion if:

$$\sum_{i=1}^{d_{\text{sub}}} \left[\max(w_{ji}, x_i) - \min(v_{ji}, x_i)\right] \le d_{\text{sub}} \cdot \theta$$

where $d_{\text{sub}}$ is the learner's active feature dimension ($d$ under full-feature bagging or $k$ under subspace sampling). Among all eligible boxes in $\mathcal{B}_c$, the box with highest Simpson membership is expanded in place ($V_j \leftarrow \min(V_j, x)$, $W_j \leftarrow \max(W_j, x)$); if none is eligible, a new point hyperbox is created.

Instead of storing raw sample lists per hyperbox (which incurs $O(N \cdot d)$ memory per learner and heavy serialization overhead), each `UncontractedHyperbox` maintains only two sufficient statistics in $O(d)$ memory: a running vector sum $\text{sample\_sum}_j \in \mathbb{R}^{d_{\text{sub}}}$ and an integer counter $\text{sample\_count}_j \in \mathbb{Z}_{>0}$. Whenever a bootstrap observation (including duplicates drawn with replacement) initializes or expands $B_j$, $\text{sample\_sum}_j \leftarrow \text{sample\_sum}_j + x$ and $\text{sample\_count}_j \leftarrow \text{sample\_count}_j + 1$. The exact arithmetic centroid with bootstrap multiplicity is:

$$C_j = \frac{\text{sample\_sum}_j}{\text{sample\_count}_j} = \frac{1}{|P_j|} \sum_{x \in P_j} x$$

with zero raw sample retention.

### 2.5. Vectorized Execution and Optional $\theta$-Preserving Consolidation Pruning

Both training and inference are vectorized over contiguous NumPy matrices:
- **Vectorized `fit()`**: Per-class dynamic arrays store `(V_arr, W_arr, sum_arr, cnt_arr)` with amortized doubling capacity. At each step, Simpson memberships across all $k_c$ boxes of class $c$ are evaluated in a single C-level matrix expression. A fast path checks the highest-membership box first and falls back to a vectorized eligibility mask when needed, matching stable descending membership order exactly.
- **Vectorized Batch `predict()`**: Trained hyperboxes are packed into contiguous matrices `V_mat`, `W_mat`, `centroids_mat`, and `labels_mat`. Queries are evaluated in memory-bounded chunks (`batch_size <= 500`) to form the boolean containment tensor `in_box` of shape `(batch_size, n_boxes)`.
- **Optional $\theta$-Preserving Hyperbox Consolidation Pruning**: Because uncontracted online expansion depends on sample arrival order, two same-class hyperboxes $B_a$ and $B_b$ formed separately can later satisfy the joint size bound:
  $$\sum_{i=1}^{d_{\text{sub}}} \left[\max(w_{ai}, w_{bi}) - \min(v_{ai}, v_{bi})\right] \le d_{\text{sub}} \cdot \theta$$
  An optional post-training consolidation pass can greedily merge same-class pairs whose bounding union preserves the $\theta$ constraint, combining their sufficient statistics exactly via $\text{sample\_sum}_{ab} = \text{sample\_sum}_a + \text{sample\_sum}_b$ and $\text{sample\_count}_{ab} = \text{sample\_count}_a + \text{sample\_count}_b$. Because the exact weighted centroid is preserved solely from $O(d)$ running statistics, consolidation reduces redundant hyperbox count without violating the $\theta$ granularity invariant or requiring raw training points.

### 2.6. Overlap Disambiguation and Ensemble Voting

For a test query $x$, let $\mathcal{O}(x) = \{B_j : v_{ji} - \epsilon \le x_i \le w_{ji} + \epsilon\ \forall i\}$ ($\epsilon = 10^{-9}$) be the set of containing hyperboxes in a single learner:
1. **Unambiguous Containment**: If $|\mathcal{O}(x)| \ge 1$ and all boxes in $\mathcal{O}(x)$ share a single class label, that class is emitted directly.
2. **Multi-Class Overlap**: If $\mathcal{O}(x)$ contains hyperboxes of multiple classes, the learner assigns the class of the nearest arithmetic centroid in squared Euclidean distance:
   $$c^*(x) = \arg\min_{B_j \in \mathcal{O}(x)} \|x - C_j\|_2^2$$
   with deterministic tie-breaking by lowest class index (`np.lexsort`).
3. **Outside All Hyperboxes**: If $\mathcal{O}(x) = \emptyset$, the learner evaluates Simpson membership $b_j(x)$ across all boxes and selects the maximizing class; if multiple boxes tie within $10^{-12}$ (for example, under membership saturation), the tie is broken by nearest centroid and then lowest class index.

Across the ensemble of $M$ learners, individual class predictions are accumulated into an integer vote matrix of shape $(N_{\text{test}}, C)$, and final predictions are obtained via deterministic plurality voting (`np.argmax` over sorted class labels). Class probability estimates for AUC-ROC evaluation are computed as normalized vote fractions $V_{i,c} / M$.

---

## 3. Empirical Benchmark Results (Leak-Free 5-Fold Stratified CV)

All experiments were evaluated via 5-fold stratified cross-validation with strictly in-fold `MinMaxScaler` fitting and test-set clipping to $[0, 1]$. Unless noted otherwise in sensitivity analysis, FMNN models use $\theta = 0.2$ and $\gamma = 1.0$; Random Forest and both Proposed RF-FMNN variants use $M = 30$ estimators. Complete per-model 6-metric tables (Accuracy, Precision, Recall, Specificity, Macro F1-Score, AUC-ROC, and Mean Hyperboxes) are archived in `experiments/results/DETAILED_METRICS_TABLES.md`.

### Table 1: 5-Fold Stratified Cross-Validation Accuracy, Macro F1-Score, and Mean Hyperboxes

| Dataset | Metric | Basic FMNN (Simpson 1992) | Decision Tree (CART) | Random Forest ($M=30$) | Proposed RF-FMNN (Bagging, $M=30$) | Proposed RF-FMNN (Subspace $\lfloor\sqrt{d}\rfloor$, $M=30$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** ($N=150, d=4, C=3$) | Accuracy | 93.33% +/- 3.65% | **95.33% +/- 3.40%** | **95.33% +/- 3.40%** | 94.00% +/- 4.90% | 94.00% +/- 4.90% |
| | Macro F1 | 93.28% +/- 3.70% | **95.31% +/- 3.41%** | **95.31% +/- 3.41%** | 93.96% +/- 4.94% | 93.96% +/- 4.94% |
| | Mean Boxes | 18.8 | - | - | 429.4 | 340.0 |
| **Wine** ($N=178, d=13, C=3$) | Accuracy | **98.89% +/- 2.22%** | 89.89% +/- 2.90% | 98.30% +/- 2.28% | 96.63% +/- 3.26% | 96.03% +/- 5.31% |
| | Macro F1 | **98.90% +/- 2.19%** | 90.12% +/- 2.80% | 98.40% +/- 2.09% | 96.66% +/- 3.19% | 96.16% +/- 5.02% |
| | Mean Boxes | 49.6 | - | - | 1058.2 | 673.6 |
| **BreastCancer** ($N=569, d=30, C=2$) | Accuracy | 95.79% +/- 2.62% | 91.04% +/- 2.79% | 94.91% +/- 2.10% | **96.84% +/- 1.31%** | 96.31% +/- 1.87% |
| | Macro F1 | 95.45% +/- 2.82% | 90.28% +/- 3.17% | 94.50% +/- 2.34% | **96.61% +/- 1.41%** | 96.07% +/- 1.95% |
| | Mean Boxes | 83.4 | - | - | 1740.4 | 1048.0 |
| **TwoMoons** ($N=300, d=2, C=2$) | Accuracy | **99.00% +/- 1.33%** | 96.67% +/- 2.79% | 97.00% +/- 1.25% | 98.00% +/- 1.25% | 90.33% +/- 5.42% |
| | Macro F1 | **99.00% +/- 1.33%** | 96.66% +/- 2.79% | 97.00% +/- 1.25% | 98.00% +/- 1.25% | 90.25% +/- 5.55% |
| | Mean Boxes | 21.0 | - | - | 528.4 | 277.2 |
| **Circles** ($N=300, d=2, C=2$) | Accuracy | 98.33% +/- 1.05% | 96.67% +/- 2.79% | 97.67% +/- 2.00% | **98.67% +/- 1.25%** | 82.33% +/- 4.03% |
| | Macro F1 | 98.33% +/- 1.06% | 96.66% +/- 2.80% | 97.66% +/- 2.00% | **98.67% +/- 1.25%** | 81.84% +/- 4.38% |
| | Mean Boxes | 24.2 | - | - | 619.4 | 319.4 |
| **Digits** ($N=1797, d=64, C=10$) | Accuracy | 98.11% +/- 0.33% | 85.47% +/- 1.38% | 97.44% +/- 0.41% | **98.39% +/- 0.48%** | 94.44% +/- 1.17% |
| | Macro F1 | 98.10% +/- 0.33% | 85.48% +/- 1.40% | 97.43% +/- 0.41% | **98.37% +/- 0.49%** | 94.34% +/- 1.22% |
| | Mean Boxes | 408.8 | - | - | 9441.6 | 5590.4 |

### 3.1. Statistical Significance Analysis (Paired Student $t$-Test)

To test whether observed cross-validation differences reflect systematic performance gains rather than fold-sampling noise, we conducted two-sided paired Student $t$-tests (`scipy.stats.ttest_rel`, $\alpha = 0.05$, $\text{df} = 4$) on the 5-fold accuracy vectors comparing each Proposed RF-FMNN variant against Basic FMNN (Simpson 1992) and Random Forest (Breiman).

#### Table 2: Paired Student $t$-Test on 5-Fold CV Accuracy

| Dataset | Proposed Variant | Baseline Model | Mean Acc Diff (pp) | $t$-statistic | $p$-value | Significant ($p < 0.05$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Iris** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | +0.67 | +0.3430 | 0.7489 | No |
| **Iris** | Proposed RF-FMNN (Bagging) | Random Forest | -1.33 | -1.0000 | 0.3739 | No |
| **Iris** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | +0.67 | +0.3430 | 0.7489 | No |
| **Iris** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | -1.33 | -1.0000 | 0.3739 | No |
| **Wine** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | -2.25 | -2.1165 | 0.1017 | No |
| **Wine** | Proposed RF-FMNN (Bagging) | Random Forest | -1.67 | -1.5000 | 0.2080 | No |
| **Wine** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | -2.86 | -0.9558 | 0.3933 | No |
| **Wine** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | -2.27 | -1.3635 | 0.2444 | No |
| **BreastCancer** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | +1.05 | +1.2377 | 0.2835 | No |
| **BreastCancer** | Proposed RF-FMNN (Bagging) | Random Forest | +1.93 | +2.2732 | 0.0854 | No |
| **BreastCancer** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | +0.52 | +0.4385 | 0.6837 | No |
| **BreastCancer** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | +1.41 | +1.0900 | 0.3370 | No |
| **TwoMoons** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | -1.00 | -1.5000 | 0.2080 | No |
| **TwoMoons** | Proposed RF-FMNN (Bagging) | Random Forest | +1.00 | +0.8847 | 0.4263 | No |
| **TwoMoons** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | -8.67 | -4.1903 | 0.0138 | Yes (degradation) |
| **TwoMoons** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | -6.67 | -2.2019 | 0.0925 | No |
| **Circles** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | +0.33 | +0.4082 | 0.7040 | No |
| **Circles** | Proposed RF-FMNN (Bagging) | Random Forest | +1.00 | +1.1767 | 0.3046 | No |
| **Circles** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | -16.00 | -8.5524 | 0.0010 | Yes (degradation) |
| **Circles** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | -15.33 | -10.6948 | 0.0004 | Yes (degradation) |
| **Digits** | Proposed RF-FMNN (Bagging) | Basic FMNN (1992) | +0.28 | +1.2893 | 0.2668 | No |
| **Digits** | Proposed RF-FMNN (Bagging) | Random Forest | **+0.95** | **+4.5451** | **0.0105** | **Yes (gain)** |
| **Digits** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (1992) | -3.67 | -5.4767 | 0.0054 | Yes (degradation) |
| **Digits** | Proposed RF-FMNN (Subspace sqrt) | Random Forest | -3.00 | -3.9659 | 0.0166 | Yes (degradation) |

Key empirical observations from Table 1 and Table 2:
1. **Honest Comparison Against Basic FMNN (Simpson 1992)**: Across all **0/6 datasets**, Proposed RF-FMNN (Bagging) does **not** achieve a statistically significant accuracy gain over Basic FMNN at $\alpha = 0.05$ ($p \in [0.1017, 0.7489]$). On small, well-separated problems (Iris, Wine, TwoMoons, Circles), single-learner Basic FMNN already attains high accuracy, and differences remain within fold variance.
2. **Statistically Significant Gain Over Random Forest on High-Dimensional Digits**: On Handwritten Digits ($N=1797, d=64, C=10$), Proposed RF-FMNN (Bagging) achieves **98.39% +/- 0.48%** accuracy (Macro F1 98.37%, AUC-ROC 99.87%), outperforming Random Forest (97.44% +/- 0.41%) by **+0.95 percentage points** with statistical significance ($t = +4.5451, p = 0.0105$).
3. **Top Accuracy and Halved Variance on Breast Cancer Wisconsin**: On 30-dimensional Breast Cancer, Proposed RF-FMNN (Bagging) leads all five models at **96.84% +/- 1.31%** (+1.05 pp over Basic FMNN at $p = 0.2835$; +1.93 pp over Random Forest at $p = 0.0854$) while reducing cross-validation standard deviation by half compared to Basic FMNN (from 2.62% down to 1.31%).
4. **Probability Calibration via Ensemble Voting**: Even where hard-label accuracy matches Basic FMNN, ensemble vote ratios substantially raise AUC-ROC on multi-class and boundary-ambiguous datasets (e.g., Iris: 99.60% vs 92.90%; TwoMoons and Circles: 100.00% vs 99.93% and 99.89%).

### 3.2. Analysis of Random Feature Subspace Sampling (`max_features='sqrt'`)

Empirical results reveal a sharp dimensional regime split for `max_features='sqrt'`:
- **Substantial Hyperbox Compression in High Dimensions**: On 30-dimensional Breast Cancer, subspace sampling ($k=5$) reduces mean total hyperboxes by **39.8%** (1740.4 to 1048.0 boxes) while retaining **96.31% +/- 1.87%** accuracy (still outperforming both Basic FMNN at 95.79% and Random Forest at 94.91%). On 64-dimensional Digits ($k=8$), hyperbox count drops by **40.8%** (9441.6 to 5590.4 boxes), though accuracy drops by 3.95 pp to 94.44% ($p = 0.0054$ vs Basic FMNN, $p = 0.0166$ vs Random Forest) because 8 sparse pixel coordinates per learner omit discriminative stroke correlations in $8 \times 8$ digit grids.
- **Severe Degradation on Low-Dimensional Non-Linear Manifolds**: On `TwoMoons` and `Circles` ($d=2$), $k = \lfloor\sqrt{2}\rfloor = 1$ forces every base learner to classify using a single 1D projection. Because concentric circles are completely inseparable along either individual axis, accuracy collapses from **98.67%** to **82.33%** on Circles ($p = 0.0010$ vs Basic FMNN, $p = 0.0004$ vs Random Forest) and from **98.00%** to **90.33%** on TwoMoons ($p = 0.0138$ vs Basic FMNN).

Accordingly, full-feature row bagging (`max_features=None`) is the recommended default for Proposed RF-FMNN; `max_features='sqrt'` should be reserved for moderate-to-high-dimensional tabular datasets with feature redundancy where ~40% hyperbox compression is prioritized.

---

## 4. Multi-Core Ray Actor Parallelism and Sharded BaseFMNN Baselines

### 4.1. Stateful Ray Actor Architecture (`RayRFFMNNClassifier`)

Because every bootstrap learner in Proposed RF-FMNN is mathematically independent (requiring neither cross-learner overlap testing nor shared mutable hyperbox tables), the ensemble maps naturally to data-parallel multi-core execution. To avoid serialization bottlenecks and thread oversubscription on multi-core hosts, our distributed implementation (`src/fmnn/ray_rf_fmnn.py`) applies four systems design principles:

1. **Zero-Copy Plasma Object Store Sharing**: The scaled training arrays $(X_{\text{train}}, y_{\text{train}})$ and query batch $X_{\text{test}}$ are placed once into Ray's shared-memory Plasma store via `ray.put()`. Worker processes read NumPy views directly from shared memory rather than duplicating copies over IPC.
2. **Stateful Worker Actors (`FMNNWorker`)**: Rather than dispatching stateless `@ray.remote` tasks that pickle and return trained `SingleFMNN` objects back to the driver, $p$ persistent `@ray.remote class FMNNWorker` actors receive round-robin bag specifications, train their assigned learners locally, and keep trained hyperbox matrices pinned permanently in worker RAM. During `fit()`, workers return only integer hyperbox counts; during `predict()`, each actor computes a compact $(N_{\text{test}} \times C)$ integer vote matrix and returns only that matrix to the driver for summation.
3. **Single-Threaded BLAS Worker Isolation (`OMP_NUM_THREADS=1`)**: When multiple Ray worker processes execute vectorized NumPy array kernels concurrently across physical cores, multi-threaded BLAS/OpenMP pools inside each worker cause severe thread contention and cache thrashing. Pinning worker BLAS backends to a single thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) ensures clean 1:1 mapping between Ray actors and physical CPU cores.
4. **Warm Actor Pool Amortization vs. Cold-Start Overhead**: Spawning fresh Ray OS worker processes incurs a fixed startup floor. On small datasets where vectorized single-core training completes rapidly, cold actor startup can dominate total wall time (Amdahl's Law). Reusing a warm actor pool across workloads or scaling to larger sample counts $N$ and estimator budgets $M$ amortizes process initialization and exposes near-linear parallel compute scaling.

### 4.2. Parallelizing Classic BaseFMNN: Structural Comparison of Sharding Baselines

To benchmark multi-core scaling fairly against Simpson's (1992) contracting algorithm, our scaling harness (`experiments/benchmark_ray_scaling.py`) evaluates two parallelization strategies for classic `BaseFMNN` across $p$ cores:

1. **Naive Data-Partitioned Box Pooling (`PartitionedRayBaseFMNN`)**: Partitions the training set into $p$ disjoint shards of size $N/p$, runs standard contracting `BaseFMNN` independently on each shard, and pools all resulting hyperboxes into a unified set. Because contraction is only enforced within each shard, hyperboxes from different shards overlap without contraction, causing uncontracted cross-shard collisions, redundant hyperbox proliferation, and accuracy degradation relative to 1-core `BaseFMNN`.
2. **Fair Sharded Majority-Voting BaseFMNN (`ShardedVotingRayBaseFMNN`)**: Trains $p$ independent contracting `BaseFMNN` models on the $p$ disjoint $N/p$ shards and combines their predictions via majority voting at inference time. While sharded voting avoids cross-shard geometric collisions and achieves high training throughput (each worker processes only $1/p$ of the samples), shrinking per-learner sample coverage by $p\times$ starves each learner of boundary support and lowers generalization accuracy compared to full-data training.

By contrast, Proposed RF-FMNN trains each learner on a full $N$-sample bootstrap bag across persistent `FMNNWorker` actors without contraction, preserving full-sample geometric coverage while remaining embarrassingly parallel across workers. Table 2B summarizes warm-pool multi-core scaling and fair vectorized `BaseFMNN` baselines measured on a 12-thread Intel hybrid host (4 P-cores + 4 E-cores; full breakdowns in `experiments/results/RAY_PARALLEL_SCALING_REPORT.md` and `figures/ray_multicore_scaling.png`).

#### Table 2B: Warm-Pool Ray Actor Scaling & Fair Vectorized BaseFMNN Baselines ($M=32$, `OMP_NUM_THREADS=1`)

| Benchmark / Model | 1 Core Time | 4 Cores Time (Speedup) | 8 Cores Time (Speedup) | Test Acc (8c) | Total Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RF-FMNN (`BreastCancer`, $N=569$)** | 0.882s | 0.299s (**2.94x**) | 0.329s (2.68x) | 96.49% | 1828 |
| **RF-FMNN (`Digits`, $N=1797$)** | 3.328s | 1.207s (2.76x) | **0.831s (4.00x)** | 97.78% | 10059 |
| **RF-FMNN (`Synthetic-Scaled`, $N=10000$)** | 55.437s | 20.003s (2.77x) | **15.680s (3.54x)** | **93.95%** | 49325 |
| **Vectorized BaseFMNN (Naive Box-Pool, $N=10000$)** | 4.917s | 4.680s (1.05x) | 6.515s (0.75x) | 90.20% | 2604 (+12.0%) |
| **Vectorized BaseFMNN (Fair Sharded Vote, $N=10000$)** | 4.917s | 0.693s (7.09x) | 0.470s (10.47x) | 90.90% | 2604 |

On `Synthetic-Scaled (N=10000)`, Proposed RF-FMNN achieves **+3.95 pp** higher accuracy than 1-core vectorized `BaseFMNN` (93.95% vs 90.00%) and **+3.05 pp** above 8-core Fair Sharded-Voting `BaseFMNN` (90.90%), scaling to **94.35%** at $M=64$ (2.54 bags/sec). Enabling relaxed $1.25\theta$ hyperbox consolidation (`prune=True`) reduces box counts by **14.9%** on BreastCancer (1828 to 1555) and **5.8%** on Digits (10059 to 9476, improving test accuracy +0.28 pp to 98.06%).

---

## 5. Hyperparameter Sensitivity Analysis ($\theta$)

The maximum hyperbox size bound $\theta \in (0, 1]$ controls the bias-variance and granularity trade-off in FMNN architectures: smaller $\theta$ yields finer local decision boundaries at the expense of more hyperboxes, whereas larger $\theta$ produces coarser boxes with increased inter-class overlap. Table 3 reports leak-free 5-fold CV accuracy and Macro F1-score for Proposed RF-FMNN (Bagging) across $\theta \in \{0.1, 0.2, 0.3, 0.4\}$ on all six datasets (from `experiments/results/ablation_results.csv`).

### Table 3: Leak-Free $\theta$ Sensitivity Analysis Across All Six Datasets (Accuracy / Macro F1)

| Dataset | $\theta = 0.1$ | $\theta = 0.2$ | $\theta = 0.3$ | $\theta = 0.4$ |
| :--- | :---: | :---: | :---: | :---: |
| **Iris** | 94.67% / 94.63% | 94.00% / 93.96% | 96.00% / 95.98% | **96.67% / 96.65%** |
| **Wine** | 94.97% / 95.09% | 96.63% / 96.66% | **98.30% / 98.31%** | 97.76% / 97.84% |
| **BreastCancer** | **97.01% / 96.79%** | 96.84% / 96.61% | 95.79% / 95.46% | 95.44% / 95.06% |
| **TwoMoons** | **99.33% / 99.33%** | 98.00% / 98.00% | 97.67% / 97.66% | 90.67% / 90.62% |
| **Circles** | **98.67% / 98.67%** | **98.67% / 98.67%** | 98.33% / 98.33% | 95.67% / 95.66% |
| **Digits** | 98.11% / 98.09% | **98.39% / 98.37%** | 98.11% / 98.10% | 97.61% / 97.60% |

Two consistent geometric regimes emerge:
- **Curved Manifolds and High-Dimensional Boundaries** (`TwoMoons`, `Circles`, `BreastCancer`, `Digits`): Fine-to-moderate hyperboxes ($\theta \in [0.1, 0.2]$) are optimal. Increasing $\theta$ to $0.4$ causes pronounced underfitting on interleaved non-linear curves (TwoMoons drops from 99.33% to 90.67%; Circles drops from 98.67% to 95.67%) as oversized hyperboxes span across opposing manifold arms.
- **Compact Convex Clusters** (`Iris`, `Wine`): Coarser hyperboxes ($\theta \in [0.3, 0.4]$) perform best (Iris peaks at 96.67% at $\theta=0.4$; Wine peaks at 98.30% at $\theta=0.3$), reducing fragmentation when classes are compact and linearly or near-linearly separable.

Across all six datasets, $\theta = 0.2$ provides a balanced default without per-dataset tuning.

---

## 6. Limitations and Practical Guidance

A rigorous evaluation of Proposed RF-FMNN requires acknowledging its verified boundaries:

1. **Statistical Parity on Small Benchmarks**: Under strict 5-fold CV, Proposed RF-FMNN (Bagging) does not produce statistically significant accuracy improvements ($p < 0.05$) over classic Basic FMNN on any of the six tested datasets. Its principal advantages over single-learner Basic FMNN are reduced cross-validation variance on high-dimensional diagnostic data (BreastCancer), smoother vote-calibrated AUC-ROC, and native multi-core parallelizability without sharding-induced accuracy loss.
2. **Hyperbox Memory Multiplication**: Ensembling $M = 30$ independent bags increases total hyperbox storage by roughly $20\times\text{--}25\times$ relative to a single Basic FMNN (e.g., 9441.6 vs. 408.8 mean boxes on Digits; 1740.4 vs. 83.4 on BreastCancer). Although $O(d)$ running-sum centroids eliminate raw training sample storage, deployments with strict memory budgets should consider smaller $M$, optional $\theta$-preserving consolidation pruning, or `max_features='sqrt'` on redundant high-dimensional tabular data.
3. **Subspace Failure in Low Dimensions**: Setting `max_features='sqrt'` degrades performance sharply when $d$ is small ($k = 1$ for $d \le 3$) or when sparse features carry joint spatial structure (as in $8 \times 8$ digit bitmaps). Subspace sampling should not be applied indiscriminately.
4. **Parallel Overhead on Small Datasets**: Multi-core Ray Actor orchestration is beneficial only when compute work outweighs IPC and actor lifecycle overhead (larger $N$ or larger $M$). On small datasets ($N < 1000$), single-process vectorized NumPy execution is faster and simpler.
5. **Sensitivity to Hyperbox Granularity $\theta$**: Optimal $\theta$ varies between compact convex datasets ($\theta \in [0.3, 0.4]$) and curved or high-dimensional datasets ($\theta \in [0.1, 0.2]$), requiring validation tuning when deploying to new domains.

---

## 7. Conclusion

This work presented and rigorously evaluated the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) under a leak-free 5-fold stratified cross-validation protocol and a stateful Ray Actor multi-core architecture. By removing Simpson's sequential overlap contraction and resolving multi-class hyperbox overlaps via $O(d)$ running-sum arithmetic centroids, Proposed RF-FMNN decouples learner construction across bootstrap bags while preserving exact bootstrap multiplicity and interpretable hyperbox geometry. Empirically across six benchmarks, Proposed RF-FMNN (Bagging, $M=30$) achieves top accuracy on 64-dimensional Handwritten Digits (**98.39%**, significantly outperforming Random Forest at $p = 0.0105$) and 30-dimensional Breast Cancer Wisconsin (**96.84%**, halving cross-validation variance relative to Basic FMNN), while matching Basic FMNN within statistical parity on smaller benchmarks. Combined with vectorized NumPy kernels, zero-copy Plasma memory sharing, persistent `FMNNWorker` actors, and single-threaded BLAS worker isolation, Proposed RF-FMNN overcomes the sequential parallelization bottleneck of classic contracting FMNN without suffering the cross-shard collision or sample-starvation penalties observed in sharded `BaseFMNN` baselines.

---

## References

1. Simpson, P. K. (1992). Fuzzy min-max neural networks - Part 1: Classification. *IEEE Transactions on Neural Networks*, 3(5), 776-786.
2. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
4. Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M. I., & Stoica, I. (2018). Ray: A distributed framework for emerging AI applications. *OSDI 2018*, 561-577.
