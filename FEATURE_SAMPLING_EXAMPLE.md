# Feature Sampling in Fuzzy Min-Max Neural Networks: Complete Step-by-Step Guide with Concrete Example

This document explains **Feature Sampling (Random Subspace Method)**: what it is, why it is used, how the number of features is calculated, and an end-to-end walkthrough with numbers.

---

## 1. What is Feature Sampling?

In machine learning, a dataset is a table with **Rows (Observations)** and **Columns (Features)**.

```
          Feature 1   Feature 2   Feature 3   Feature 4  ...  Feature d    |  Class Label
Obs 1       0.25        0.81        0.12        0.67             0.44     |      0
Obs 2       0.73        0.15        0.89        0.32             0.91     |      1
Obs 3       0.41        0.52        0.63        0.18             0.29     |      0
```

- **Row Sampling (Bagging):** Selecting random rows (samples) with replacement.
- **Feature Sampling (Subspaces):** Selecting random columns (features) without replacement for each base FMNN learner in the ensemble.

### Supported RF-FMNN Operating Modes

The RF-FMNN implementation supports two feature-selection modes controlled via `max_features`:

- **Row-bagging mode (`max_features=None`):** Each base FMNN learner uses all $d$ features. Every learner differs only in its bootstrap row sample.
- **Feature-subspace mode (`max_features='sqrt'`):** Each base FMNN learner uses a random subset of $k = \lfloor\sqrt{d}\rfloor$ features sampled without replacement alongside its bootstrap row sample.

---

## 2. Why Do We Need Feature Sampling in Hyperbox Networks?

### A. The "Curse of Dimensionality" in Hyperbox Geometries
A hyperbox is a multidimensional box. Its volume is the product of its side lengths:

$$\text{Volume}(B_j) = \prod_{i=1}^d (w_{ji} - v_{ji})$$

When the number of features $d$ is large (e.g., $d = 30$ in Breast Cancer, or $d = 60$ in Sonar):
- If the average box width is $0.2$, the volume in 30 dimensions is $0.2^{30} \approx 10^{-21}$ (practically zero).
- In high-dimensional space, almost all hyperboxes become empty pinpricks.
- The network is forced to create a separate hyperbox for almost every individual training point. This causes memorization (overfitting) and slow inference.

### B. Decorrelating Base Classifiers (The Forest Principle)
If every model in the ensemble looks at the exact same 30 features, they will all make the same mistakes.
By forcing Model 1 to look at Features `[1, 5, 8]` and Model 2 to look at Features `[2, 3, 9]`:
- Model 1 learns patterns invisible to Model 2.
- Individual mistakes cancel out during majority voting.
- Ensemble variance drops: $\text{Var}_{\text{ensemble}} = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$ (lower correlation $\rho \implies$ lower overall error).

---

## 3. How Many Features Do We Choose ($k$) and Why Exactly That Number?

### The Mathematical Formula:

For classification problems, the standard choice established by Leo Breiman (2001) is:

$$k = \lfloor\sqrt{d}\rfloor$$

where $d$ is the total number of features in the dataset.

Alternative options:
- Logarithmic: $k = \lfloor\log_2(d) + 1\rfloor$ (used for very high dimensions $d > 500$)
- Percentage: $k = \lfloor 0.5 \times d \rfloor$ (half the features)

### Why Exactly $\sqrt{d}$?
1. **Balance between diversity and signal:**
   - If $k$ is too small ($k=1$), a single feature does not carry enough information to separate classes (high bias / weak individual learners).
   - If $k$ is too large ($k=d$), all learners rely on the same dominant features, increasing inter-learner correlation and limiting variance reduction.
   - Breiman (2001) established $k = \lfloor\sqrt{d}\rfloor$ as the standard classification default that balances individual learner strength ($s$) against inter-learner correlation ($\bar{\rho}$) in the ensemble generalization error bound $PE^* \le \bar{\rho}(1 - s^2) / s^2$.
2. **Dimension reduction for hyperboxes:**
   - For $d = 30$ features: $k = \lfloor\sqrt{30}\rfloor = 5$ features.
   - Building hyperboxes in 5 dimensions instead of 30 dimensions prevents volume collapse and keeps hyperboxes compact and interpretable.

---

## 4. Do All Features Occur, or Are Some Missed?

### Within a Single Base Learner:
Yes, in feature-subspace mode (`max_features='sqrt'`), a single base FMNN learner **deliberately omits** $(d - k)$ features. Each learner is intentionally constrained so it learns a distinct low-dimensional projection.

### Across the Complete Ensemble of $M$ Learners:
No feature is permanently missed. Because each learner samples $k$ features out of $d$ without replacement independently, the probability that a specific feature is **never** picked in any of the $M$ learners is:

$$P(\text{Feature never picked}) = \left( 1 - \frac{k}{d} \right)^M$$

### Real Calculation:
- Dataset: $d = 30$ features, $k = 5$ sampled features per learner (`max_features='sqrt'`).
- Ensemble: $M = 35$ base FMNN learners.
- Fraction picked per learner: $5 / 30 = 16.7\%$.
- Probability a feature is missed across all 35 learners:
  $$P(\text{missed}) = \left(1 - \frac{5}{30}\right)^{35} = \left(\frac{5}{6}\right)^{35} \approx 0.0017 \quad (0.17\%)$$

Across the full ensemble, **every feature appears multiple times** (on average, each feature appears in $35 \times \frac{5}{30} \approx 5.8$ learners).

Note on row coverage: independently of feature selection, RF-FMNN generates bootstrap bags of size $N$ (with replacement) until all $N$ training observations are covered at least once (Coupon-Collector rule). Because each bag performs $N$ draws, the expected number of **bags** required for complete coverage is $\mathbb{E}[B] \approx \ln(N) + \gamma$ (where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant), corresponding to $\approx N(\ln N + \gamma)$ total individual sample draws across those bags.

---

## 5. Complete Step-by-Step Walkthrough with Numbers

Let us take a concrete medical diagnostic problem: **Predicting Tumor Type (Benign = 0, Malignant = 1)**.

### Input Dataset
- Number of features: $d = 9$ numerical measurements, normalized to $[0, 1]$ using `MinMaxScaler` fitted strictly on the training fold inside cross-validation (with test points transformed using training bounds and clipped to $[0, 1]$ to prevent data leakage):
  1. $F_1$: Radius Mean
  2. $F_2$: Texture Mean
  3. $F_3$: Perimeter Mean
  4. $F_4$: Area Mean
  5. $F_5$: Smoothness Mean
  6. $F_6$: Compactness Mean
  7. $F_7$: Concavity Mean
  8. $F_8$: Symmetry Mean
  9. $F_9$: Fractal Dimension

---

### Step 1: Calculate $k$ (Feature-Subspace Mode, `max_features='sqrt'`)
$$k = \lfloor\sqrt{9}\rfloor = 3 \text{ features per base FMNN learner}$$

Every independent FMNN learner builds 3-dimensional hyperboxes instead of 9-dimensional boxes. (In row-bagging mode with `max_features=None`, every learner uses all $d = 9$ features.)

---

### Step 2: Random Feature Subspace Selection

Suppose our ensemble has $M = 3$ base FMNN learners for illustration:
- **Learner 1** randomly selects 3 features: $\{F_1, F_4, F_7\}$ (Radius, Area, Concavity)
- **Learner 2** randomly selects 3 features: $\{F_2, F_5, F_8\}$ (Texture, Smoothness, Symmetry)
- **Learner 3** randomly selects 3 features: $\{F_3, F_6, F_9\}$ (Perimeter, Compactness, Fractal Dim)

Notice:
- No learner looks at all 9 features.
- Together, all 9 features $\{F_1, \dots, F_9\}$ are fully covered.

---

### Step 3: Training Hyperboxes in Feature Subspaces

Assume maximum hyperbox size parameter $\theta = 0.2$.

#### For Learner 1:
Learner 1 filters its bootstrap training rows to keep only columns $[F_1, F_4, F_7]$.
Suppose training sample $x_1 = [F_1=0.20, F_4=0.15, F_7=0.10]$ arrives with **Class 0 (Benign)**:
- Learner 1 creates 3D `UncontractedHyperbox` $B_1$:
  - Min boundary: $V_1 = [0.20, 0.15, 0.10]$
  - Max boundary: $W_1 = [0.20, 0.15, 0.10]$
  - Running sum & count: `sample_sum` $= [0.20, 0.15, 0.10]$, `sample_count` $= 1$
  - Centroid: $C_1 = \text{sample\_sum} / \text{sample\_count} = [0.20, 0.15, 0.10]$

A second training sample of Class 0 arrives: $x_2 = [0.25, 0.18, 0.12]$:
- Expansion check ($\sum_{i=1}^k (\max(w_{1i}, x_i) - \min(v_{1i}, x_i)) \le k \cdot \theta$ with $k = 3, \theta = 0.2$):
  $$\Delta = (0.25 - 0.20) + (0.18 - 0.15) + (0.12 - 0.10) = 0.05 + 0.03 + 0.02 = 0.10 \le 3 \times 0.2 = 0.60$$
- Fits! Box $B_1$ expands without contraction:
  - $V_1 = [0.20, 0.15, 0.10]$
  - $W_1 = [0.25, 0.18, 0.12]$
  - Running sum & count: `sample_sum` $= [0.45, 0.33, 0.22]$, `sample_count` $= 2$
  - Centroid: $C_1 = \text{sample\_sum} / \text{sample\_count} = [0.225, 0.165, 0.110]$

Because `UncontractedHyperbox` updates `sample_sum` and `sample_count` incrementally, duplicate observations drawn in a bootstrap bag contribute according to their exact multiplicity while requiring only $O(k)$ (or $O(d)$ in full-feature mode) memory per box without retaining raw training copies.

Learner 2 and Learner 3 perform the same uncontracted expansion in parallel for their respective 3-feature subspaces.

---

### Step 4: Classifying an Unseen Test Observation

Now a new patient arrives with complete 9-feature reading (normalized using the training fold's `MinMaxScaler` and clipped to $[0, 1]$):
$$x_{test} = [0.22, 0.65, 0.40, 0.16, 0.70, 0.35, 0.11, 0.80, 0.25]$$

How does the ensemble classify this patient?

#### 1. Evaluation in Learner 1:
- Extracts its assigned features $\{F_1, F_4, F_7\}$ from $x_{test}$:
  $$x_{test}^{(1)} = [0.22, 0.16, 0.11]$$
- Check against Box $B_1$ ($V_1 = [0.20, 0.15, 0.10]$, $W_1 = [0.25, 0.18, 0.12]$):
  - $0.20 \le 0.22 \le 0.25$ (True)
  - $0.15 \le 0.16 \le 0.18$ (True)
  - $0.10 \le 0.11 \le 0.12$ (True)
- $x_{test}^{(1)}$ is inside Box $B_1$ of Class 0.
- **Learner 1 Vote: Class 0 (Benign)**

#### 2. Evaluation in Learner 2:
- Extracts its assigned features $\{F_2, F_5, F_8\}$ from $x_{test}$:
  $$x_{test}^{(2)} = [0.65, 0.70, 0.80]$$
- Suppose $x_{test}^{(2)}$ lands inside Box $B_5$ of Class 1:
- **Learner 2 Vote: Class 1 (Malignant)**

#### 3. Evaluation in Learner 3:
- Extracts its assigned features $\{F_3, F_6, F_9\}$ from $x_{test}$:
  $$x_{test}^{(3)} = [0.40, 0.35, 0.25]$$
- Suppose $x_{test}^{(3)}$ falls outside all boxes, and fuzzy membership is highest for Box $B_2$ of Class 0 ($b = 0.88$):
- **Learner 3 Vote: Class 0 (Benign)**

---

### Step 5: Final Ensemble Decision

Collect votes from all learners:
- Learner 1: **Class 0**
- Learner 2: **Class 1**
- Learner 3: **Class 0**

Tally:
- Class 0 (Benign): **2 votes**
- Class 1 (Malignant): **1 vote**

**Final Prediction: Class 0 (Benign)**.

---

## 6. Summary Comparison Table

| Attribute | Row-Bagging Mode (`max_features=None`, $k = d$) | Feature-Subspace Mode (`max_features='sqrt'`, $k = \lfloor\sqrt{d}\rfloor$) |
| :--- | :--- | :--- |
| **Hyperbox Dimensions** | Full space (e.g. 30D or 60D) | Low-dimensional subspace (e.g. 5D or 7D) |
| **Hyperbox Volume** | Collapses toward zero in high dimensions ($\approx 0$) | Stays compact and well-populated |
| **Number of Boxes** | Large in high dimensions (fragmentation into near-singleton boxes) | Moderate (meaningful cluster boxes) |
| **Learner Correlation** | Higher (learners share identical feature axes) | Lower (learners make decorrelated errors) |
| **Best Use Case** | Low-dimensional datasets where every feature carries essential signal | Medium-to-high-dimensional or noisy datasets prone to volume collapse |
