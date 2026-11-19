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
2. It calculates the **center point (centroid)** of the real data observations inside each hyperbox.
3. If a test observation falls inside overlapping hyperboxes of different classes, classification is determined by the **nearest data-based centroid**.
4. It trains **an ensemble** of multiple independent FMNN models using bootstrap random sampling with replacement, then uses majority voting to make the final classification.

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
| STEP 1: PREPROCESSING & SCALING                                               |
| Check missing values, encode labels, and normalize features to [0.0, 1.0]     |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 2: DATASET SPLIT                                                         |
| Training Set (used for sampling & model training) | Test Set (evaluation)     |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 3: BOOTSTRAP RANDOM SAMPLING                                             |
| Generate bootstrap samples with replacement until the union of unique IDs     |
| covers 100% of the training dataset observations                              |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 4: INDEPENDENT FMNN TRAINING (WITHOUT CONTRACTION)                       |
|   - Provide each bootstrap sample to a separate FMNN model                    |
|   - Expand hyperboxes according to expansion criteria and max size theta      |
|   - Do not contract or modify overlapping hyperboxes                          |
|   - Calculate hyperbox centroid as arithmetic mean of assigned observations   |
|     (multiplicity of duplicate observations is retained)                      |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 5: TEST OBSERVATION EVALUATION                                           |
|   - Case A (Inside Overlapping Boxes of Diff Classes): Resolve via nearest    |
|     data-based centroid using Euclidean distance                              |
|   - Case B (Inside Single Box / Non-overlapping): Assign that box's class     |
|   - Case C (Outside All Boxes): Use FMNN fuzzy membership function            |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 6: ENSEMBLE MAJORITY VOTING                                              |
| Collect independent predictions from all FMNN models                          |
| Select class receiving the highest number of votes as final prediction        |
+-------------------------------------------------------------------------------+
```

---

## 3. How Bootstrap Sampling Works

Bootstrap sampling is performed with replacement on the training dataset.

### Sampling With Replacement
Imagine a dataset with 5 training observations: `[Obs 1, Obs 2, Obs 3, Obs 4, Obs 5]`.
When creating a bootstrap sample of size 5:
1. Draw an observation at random, record its ID, and **return it to the pool**.
2. Repeat 5 times.

A single bootstrap sample might contain:
`Sample 1: [Obs 2, Obs 2, Obs 4, Obs 1, Obs 5]`  (Obs 3 was omitted, Obs 2 appears twice)

### The Complete Coverage Rule
In the proposed RF-FMNN:
- Bootstrap samples are continuously generated until the union of their unique observation IDs covers the entire original training dataset:
  $$\bigcup_{b=1}^B I_b = \{1, 2, \dots, N\}$$
- This ensures that every original training observation contributes to at least one FMNN classifier in the ensemble.

---

## 4. Hyperbox Construction and Overlap Handling

### Hyperbox Expansion
A hyperbox $B_j$ is defined by:
- $V_j$: Lower boundary vector $(v_{j1}, \dots, v_{jn})$
- $W_j$: Upper boundary vector $(w_{j1}, \dots, w_{jn})$

When a training observation $x$ of class $c$ is evaluated:
1. Search existing hyperboxes of class $c$.
2. Check if expanding box $B_j$ to include $x$ satisfies maximum size $\theta$:
   $$\sum_{i=1}^n (\max(w_{ji}, x_i) - \min(v_{ji}, x_i)) \le n \cdot \theta$$
3. If satisfied, expand:
   $$v_{ji} \leftarrow \min(v_{ji}, x_i), \quad w_{ji} \leftarrow \max(w_{ji}, x_i)$$
4. If no existing box of class $c$ can expand, initialize a new hyperbox with $V = x, W = x$.

### Centroid Calculation
The centroid $C_j$ is computed as the arithmetic mean of all observations assigned to that box:

$$C_j = \frac{1}{|P_j|} \sum_{x \in P_j} x$$

If an observation was drawn multiple times in the bootstrap sample, it is counted according to its multiplicity, correctly biasing the centroid toward denser regions.

### Overlap Handling Without Contraction
Unlike conventional FMNN approaches that use overlap contraction, RF-FMNN preserves the learned hyperbox boundaries.

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
- The class associated with the nearest centroid is selected.

---

## 5. From Training to Testing: Step-by-Step Data Journey

### Stage 1: Data Preprocessing
- Check missing values and encode categorical features.
- Scale numerical features into $[0.0, 1.0]$ using Min-Max normalization.

### Stage 2: Training Pipeline
1. Split data into Training and Testing portions.
2. Generate bootstrap samples until all observation IDs are covered.
3. Train an independent FMNN model on each bootstrap sample using hyperbox expansion.
4. Calculate and store data-based centroids for all hyperboxes.

### Stage 3: Testing Pipeline
For each test observation $x_{test}$:
1. Pass $x_{test}$ to each independent FMNN model:
   - If $x_{test}$ is inside overlapping hyperboxes of different classes -> assign class of nearest centroid.
   - If $x_{test}$ is inside hyperbox(es) of one class -> assign that class.
   - If $x_{test}$ is outside all hyperboxes -> compute FMNN membership $b_j(x)$ to find highest membership.
2. Collect predictions from all models.
3. Perform majority/plurality voting to produce final classification.

---

## 6. Architecture Component Summary

| Component | Function | Role in RF-FMNN |
| :--- | :--- | :--- |
| **Input & Preprocessing** | Normalizes data into $[0.0, 1.0]$ | Ensures input values match FMNN unit hypercube requirements |
| **Bootstrap Sampling Module** | Generates samples with replacement until complete ID coverage | Provides independent data subsets while ensuring all training observations contribute |
| **Independent FMNN Classifiers** | Constructs class-labelled hyperboxes per sample | Forms independent spatial representations without cross-model interference |
| **Hyperbox Construction & Expansion** | Expands boundaries up to size limit $\theta$ | Clusters observations without applying overlap contraction |
| **Hyperbox Centroid Module** | Computes arithmetic mean of assigned observations | Captures the true spatial center of mass for each hyperbox |
| **Overlap Resolution Module** | Computes Euclidean distance to candidate centroids | Resolves classification conflict in overlapping regions |
| **Ensemble Voting Module** | Aggregates individual model predictions | Applies majority voting for final, robust classification output |
