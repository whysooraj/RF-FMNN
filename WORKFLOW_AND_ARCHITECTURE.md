# RF-FMNN: Workflow, Architecture, and Data Lifecycle Explained

A plain-language, step-by-step guide explaining how the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) works, from raw training data to final test predictions.

---

## 1. What is RF-FMNN in Plain English?

Traditional machine learning algorithms draw lines, curves, or decision trees to separate categories.
A **Fuzzy Min-Max Neural Network (FMNN)** does something different: it draws **boxes** (hyperboxes) around groups of data points in feature space.

- If a point is **inside** a box, it belongs to that box's class.
- If a point is **outside** all boxes, it calculates how close it is to the nearest box surface using a **fuzzy membership score** between 0 and 1.

### The Problem with Conventional FMNN:
When two boxes of different classes touch or overlap, the original 1992 algorithm cuts and shrinks them (called *Simpson's contraction*). This cuts away real data regions and causes errors in complex, overlapping, or high-dimensional spaces.

### The RF-FMNN Methodology:
Instead of cutting overlapping boxes:
1. It **keeps the hyperbox boundaries intact** without contraction.
2. It tracks the **arithmetic mean (centroid)** of assigned training observations inside each `UncontractedHyperbox` using a running sum and count (`sample_sum / sample_count`) in $O(d)$ memory without retaining raw training copies.
3. If a test observation falls inside overlapping hyperboxes of different classes, classification is determined by the **nearest data-based centroid**.
4. It trains **an ensemble** of independent FMNN learners using bootstrap random sampling with replacement (in either row-bagging mode `max_features=None` with full $d$ features, or feature-subspace mode `max_features='sqrt'` with $k = \lfloor\sqrt{d}\rfloor$ features), optionally parallelized via Stateful Ray Actors (`FMNNWorker`), and aggregates votes via majority/plurality voting.

---

## 2. End-to-End Workflow Diagram

```
+-------------------------------------------------------------------------------+
|                             RAW DATASET                                       |
|                    Features: Numerical Values | Label: Class Label            |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 1: DATASET / CROSS-VALIDATION FOLD SPLIT                                 |
| Split into Training Fold and Test Fold BEFORE fitting any scaler              |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 2: LEAKAGE-FREE PREPROCESSING & SCALING                                  |
| - Fit MinMaxScaler strictly on the Training Fold -> [0.0, 1.0]^d              |
| - Transform Test Fold using Training Fold bounds and clip to [0.0, 1.0]       |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 3: BOOTSTRAP SAMPLING & FEATURE MODE SELECTION                           |
| - Draw bags of N samples with replacement until 100% training ID coverage     |
|   (Coupon-Collector expected bags: E[B] ~ ln(N) + gamma, gamma ~ 0.5772)      |
| - Mode: max_features=None (full d features) OR max_features='sqrt'            |
|   (k = floor(sqrt(d)) features per learner without replacement)               |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 4: INDEPENDENT FMNN TRAINING (NO CONTRACTION, RAY ACTOR PARALLELISM)     |
| - Place (X_train, y_train) once into Ray Plasma Object Store via ray.put()    |
| - Stateful Ray Actors (@ray.remote class FMNNWorker) train assigned bags      |
| - Expand UncontractedHyperbox up to max size theta (NO overlap contraction)   |
| - Maintain running sample_sum / sample_count in O(d) memory (exact bootstrap  |
|   multiplicity, no raw training copies retained)                              |
| - Pin trained SingleFMNN models inside worker RAM (zero pickling overhead)    |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 5: TEST OBSERVATION EVALUATION (ZERO MODEL PICKLING)                     |
| - Share clipped X_test via ray.put(); FMNNWorker actors evaluate pinned models|
|   - Case A (Inside Overlapping Boxes of Diff Classes): Nearest data centroid  |
|     using Euclidean distance                                                  |
|   - Case B (Inside Single Box / Same-Class Boxes): Assign that class          |
|   - Case C (Outside All Boxes): Use Simpson FMNN fuzzy membership function    |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 6: ENSEMBLE MAJORITY VOTING                                              |
| Gather compact prediction arrays from all learners/actors                     |
| Select class receiving the highest number of votes as final prediction        |
+-------------------------------------------------------------------------------+
```

---

## 3. How Bootstrap Sampling and Feature Modes Work

Bootstrap sampling is performed with replacement on the training fold.

### Sampling With Replacement
Imagine a training fold with 5 observations: `[Obs 1, Obs 2, Obs 3, Obs 4, Obs 5]`.
When creating a bootstrap bag of size $N = 5$:
1. Draw an observation at random, record its ID, and **return it to the pool**.
2. Repeat $N = 5$ times.

A single bootstrap bag might contain:
`Bag 1: [Obs 2, Obs 2, Obs 4, Obs 1, Obs 5]` (Obs 3 was omitted, Obs 2 appears twice).

### The Complete Coverage Rule (Coupon-Collector Expectation)
In the proposed RF-FMNN:
- Bootstrap bags (each drawing $N$ samples with replacement) are generated until the union of their unique observation IDs covers the entire training fold (up to an optional safety cap `max_bags_limit`, or a fixed count when `n_estimators` is specified):
  $$\bigcup_{b=1}^B I_b = \{1, 2, \dots, N\}$$
- This guarantees that every training observation contributes to at least one FMNN learner in the ensemble.
- Because each bag draws $N$ samples with replacement, the expected number of **bags** required to cover all $N$ observations is:
  $$\mathbb{E}[B] \approx \ln(N) + \gamma$$
  where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant. Note that $\approx N(\ln N + \gamma)$ is the expected total number of *individual sample draws* across those bags, not the number of bags.

### Two Supported Feature Modes
RF-FMNN supports two feature-selection modes controlled by `max_features`:
- **Row-bagging mode (`max_features=None`)**: Each base FMNN learner trains on all $d$ features. Diversity comes strictly from bootstrap row resampling.
- **Feature-subspace mode (`max_features='sqrt'`)**: Each base FMNN learner trains on $k = \lfloor\sqrt{d}\rfloor$ features sampled without replacement. This reduces hyperbox dimensionality in high-dimensional datasets and decorrelates base learners (see `FEATURE_SAMPLING_EXAMPLE.md` for a step-by-step numerical walkthrough).

---

## 4. Hyperbox Construction, Memory-Efficient Centroids, and Overlap Handling

### Hyperbox Expansion
Let $k$ denote the active feature count per learner ($k = d$ for `max_features=None`; $k = \lfloor\sqrt{d}\rfloor$ for `max_features='sqrt'`). An `UncontractedHyperbox` $B_j$ is defined by:
- $V_j$: Lower boundary vector $(v_{j1}, \dots, v_{jk})$
- $W_j$: Upper boundary vector $(w_{j1}, \dots, w_{jk})$

When a training observation $x$ of class $c$ is evaluated:
1. Search existing hyperboxes of class $c$ (ordered by descending fuzzy membership).
2. Check if expanding box $B_j$ to include $x$ satisfies maximum size $\theta$:
   $$\sum_{i=1}^k (\max(w_{ji}, x_i) - \min(v_{ji}, x_i)) \le k \cdot \theta$$
3. If satisfied, expand:
   $$v_{ji} \leftarrow \min(v_{ji}, x_i), \quad w_{ji} \leftarrow \max(w_{ji}, x_i)$$
4. If no existing box of class $c$ can expand, initialize a new hyperbox with $V = x, W = x$.

### Centroid Calculation ($O(d)$ Running Sum & Count)
Mathematically, the centroid $C_j$ is the arithmetic mean of the multiset $P_j$ of all observations assigned to hyperbox $B_j$:

$$C_j = \frac{1}{|P_j|} \sum_{x \in P_j} x = \frac{\text{sample\_sum}_j}{\text{sample\_count}_j}$$

In implementation, `UncontractedHyperbox` stores only a running arithmetic sum vector (`sample_sum`) and an integer observation counter (`sample_count`) rather than appending raw training arrays. Every time an observation is assigned to $B_j$ (including repeated draws in a bootstrap bag), `sample_sum += x` and `sample_count += 1`. This preserves exact bootstrap multiplicity while using only $O(k)$ (up to $O(d)$) memory per hyperbox and keeping zero raw training copies in memory.

### Overlap Handling Without Contraction
Unlike conventional FMNN approaches that apply Simpson's 4-case overlap contraction, RF-FMNN preserves the learned hyperbox boundaries.

```
+---------------------------------------+
|  Hyperbox A (Class 0)                 |
|                                       |
|          [Centroid A]                 |
|                   +-------------------+-------------------+
|                   | Overlap Zone      |                   |
|                   |       x (Test)    |                   |
|                   |                   |    [Centroid B]   |
+-------------------+-------------------+                   |
                    |                     Hyperbox B (Class 1)
                    +---------------------------------------+
```

When a test observation $x$ falls inside overlapping hyperboxes of different classes:
- Calculate Euclidean distance to Centroid A: $D_A = \|x - C_A\|_2$
- Calculate Euclidean distance to Centroid B: $D_B = \|x - C_B\|_2$
- Select the class associated with the nearest centroid.

---

## 5. Distributed Execution with Stateful Ray Actors

For multi-core scaling, `RayRFFMNNClassifier` distributes training and inference without serializing trained models across processes:

1. **Zero-Copy Shared Memory (`ray.put`)**: The normalized training arrays $(X_{\text{train}}, y_{\text{train}})$ are placed once into Ray's Plasma Object Store so worker processes read shared memory without redundant copies.
2. **Stateful Ray Actors (`@ray.remote class FMNNWorker`)**: A pool of persistent worker actors receives partitions of bootstrap bag indices (and feature subspace indices when `max_features='sqrt'`). Each actor trains its assigned `SingleFMNN` learners in parallel and **pins the trained hyperbox models inside worker RAM**.
3. **Zero Model Pickling Between `fit()` and `predict()`**: At inference time, only `ray.put(X_test)` is shared with the actors. Each `FMNNWorker` evaluates its pinned models locally and returns only compact prediction vote arrays, avoiding model pickling overhead completely.

---

## 6. From Training to Testing: Step-by-Step Data Journey

### Stage 1: Leakage-Free Fold Split and Normalization
1. Split the dataset into Training and Testing folds (e.g., stratified $k$-fold cross-validation).
2. Fit `MinMaxScaler` strictly on the Training Fold to map training features into $[0.0, 1.0]$.
3. Transform the Testing Fold using the Training Fold's scaler parameters and clip values to $[0.0, 1.0]$ so test-set extrema never leak into training normalization while preserving FMNN unit-hypercube bounds.

### Stage 2: Training Pipeline
1. Generate bootstrap bags of size $N$ with replacement until complete training ID coverage ($\mathbb{E}[B] \approx \ln(N) + \gamma$ bags) or fixed `n_estimators`.
2. Assign either all $d$ features (`max_features=None`) or $k = \lfloor\sqrt{d}\rfloor$ random features (`max_features='sqrt'`) to each learner.
3. Train independent `SingleFMNN` learners (sequentially or in parallel inside stateful `FMNNWorker` Ray actors reading from Plasma via `ray.put`).
4. Finalize hyperbox centroids via `sample_sum / sample_count` and pin models in worker RAM (Ray mode).

### Stage 3: Testing Pipeline
For each normalized, clipped test observation $x_{test}$:
1. Evaluate $x_{test}$ (projected onto each learner's feature subset when `max_features='sqrt'`) across all independent FMNN learners (executed locally on pinned actor models in Ray mode):
   - If $x_{test}$ is inside overlapping hyperboxes of different classes -> assign the class of the nearest centroid.
   - If $x_{test}$ is inside hyperbox(es) of a single class -> assign that class.
   - If $x_{test}$ is outside all hyperboxes -> compute Simpson FMNN membership $b_j(x)$ and select the class with highest membership.
2. Gather compact prediction arrays from all learners/actors.
3. Perform majority/plurality voting (and vote-fraction probability estimation for `predict_proba`) to produce the final classification.

---

## 7. Architecture Component Summary

| Component | Function | Role in RF-FMNN |
| :--- | :--- | :--- |
| **Fold Split & Leakage-Free Scaler** | Fits `MinMaxScaler` on training fold only; clips test fold to $[0.0, 1.0]$ | Satisfies FMNN unit-hypercube bounds without test-set data leakage |
| **Bootstrap & Feature Sampling** | Draws bags until full ID coverage ($\mathbb{E}[B] \approx \ln(N) + \gamma$ bags); supports `max_features=None` ($d$) and `'sqrt'` ($\lfloor\sqrt{d}\rfloor$) | Guarantees complete training coverage and controls feature-space diversity |
| **Stateful Ray Actors (`FMNNWorker`)** | Reads shared data from Plasma (`ray.put`), trains assigned bags, pins models in worker RAM | Enables parallel training and zero-pickling inference across CPU cores |
| **Uncontracted Hyperbox (`UncontractedHyperbox`)** | Expands boundaries up to size limit $\theta$ without overlap contraction | Clusters observations without geometric clipping or order-dependent shrinkage |
| **Running Centroid Accumulator** | Tracks `sample_sum` and `sample_count` ($C_j = \text{sample\_sum} / \text{sample\_count}$) | Preserves exact bootstrap multiplicity in $O(d)$ memory without retaining raw training copies |
| **Overlap Resolution Module** | Computes Euclidean distance to candidate centroids in multi-class overlaps | Resolves classification conflicts in overlapping regions using data density |
| **Ensemble Voting Module** | Aggregates compact vote arrays across all base learners | Applies plurality/majority voting for stable final predictions |
