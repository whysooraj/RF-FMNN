# RF-FMNN: Workflow, Architecture, and Data Lifecycle Explained

A plain-language, step-by-step guide explaining how the Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN) works, from raw training data to final test predictions.

---

## 1. What is RF-FMNN in Plain English?

Traditional machine learning algorithms draw lines, curves, or decision trees to separate categories.
A **Fuzzy Min-Max Neural Network (FMNN)** does something different: it draws **boxes** (hyperboxes) around groups of data points in feature space.

- If a point is **inside** a box, it belongs to that box's class.
- If a point is **outside** all boxes, it calculates how close it is to the nearest box surface using a **fuzzy membership score** between 0 and 1.

### The Big Problem with Old FMNN:
When two boxes of different classes touch or overlap, the old 1992 algorithm cuts and shrinks them (called  *Simpsons comtraaction*). This cuts away real data regions and causes errors in high dimensions.

### The RF-FMNN Solution:
Instead of cutting overlapping boxes:
1. It **keeps the boxes intact** without shrinking them.
2. It calculates the **center point (centroid)** of the real data points inside each box.
3. If an input falls inside overlapping boxes from different classes, it awards the win to whichever box has its **centroid closest** to that point.
4. It trains **an ensemble (forest)** of these box networks using random samples (bootstrap) and random features (subspaces), then votes on the final answer.

---

## 2. End-to-End Workflow Diagram

```
+-------------------------------------------------------------------------------+
|                             RAW DATASET (e.g. Iris)                           |
|                    Features: Length, Width | Label: Flower Type               |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 1: PREPROCESSING & SCALING                                               |
| Scale every feature strictly into range [0.0, 1.0] using Min-Max scaling      |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 2: DATASET SPLIT                                                         |
| 80% Training Data (used to build boxes) | 20% Test Data (hidden until evaluation)|
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 3: ENSEMBLE SAMPLING (For each Tree m = 1 to M):                         |
|   A. Row Sampling (Bootstrap with replacement: pick N rows randomly)          |
|   B. Column Sampling (Subspace: pick k = sqrt(d) features randomly)           |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 4: HYPERBOX EXPANSION & CENTROID CALCULATION                             |
|   - Feed sampled points one by one                                            |
|   - Grow existing box if expanded size <= max size (theta)                    |
|   - Otherwise create new box                                                  |
|   - DO NOT contract overlapping boxes                                         |
|   - Calculate average position (centroid) of all points inside each box       |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 5: TEST INFERENCE (For an unseen test sample x):                         |
|   - Each Tree m looks at sample x using its selected features                 |
|   - If x is inside boxes of competing classes: use nearest centroid           |
|   - If x is outside all boxes: use fuzzy boundary membership                  |
|   - Each Tree produces a class prediction or fuzzy score vector               |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| STEP 6: ENSEMBLE CONSENSUS (VOTING)                                           |
| Combine outputs across all M trees (majority vote or soft fuzzy sum)          |
| Output: Final Class Label                                                     |
+-------------------------------------------------------------------------------+
```

---

## 3. How Sampling Works (Row & Column Sampling)

Sampling is the engine that gives the ensemble its stability and power. It consists of two stages: **Row Sampling (Bagging)** and **Column Sampling (Random Subspaces)**.

### A. Row Sampling (Bootstrap with Replacement)

Imagine an urn with 5 balls numbered `[1, 2, 3, 4, 5]`.
To create a bootstrap sample of size 5:
1. Draw a ball, write down its number, and **put it back in the urn**.
2. Repeat 5 times.

You might end up with:
`Bag 1: [2, 2, 4, 1, 5]`  (Ball 3 was missed, Ball 2 appeared twice)

In machine learning terms:
- **In-Bag Data (~63.2%)**: The unique training points chosen for that tree.
- **Duplicates**: Some points appear 2 or 3 times. When calculating the centroid of a hyperbox, duplicate points are counted with their full multiplicity (weighting the centroid toward denser regions).
- **Out-of-Bag (OOB) Data (~36.8%)**: Points never picked for this tree. They act as an automatic validation set to score the model without touching the test set.

### B. Column Sampling (Random Feature Subspaces)

If the dataset has 30 features (e.g. Breast Cancer):
- A single tree does not look at all 30 features at once.
- Instead, each tree randomly selects a subset of features:
  $$k = \lfloor\sqrt{30}\rfloor \approx 5 \text{ features}$$

**Why this matters:**
In 30 dimensions, hyperboxes suffer from the curse of dimensionality (they become empty shells). By projecting down to 5 random features per tree:
1. Hyperboxes remain compact and capture meaningful cluster densities.
2. Different trees learn different perspectives of the data (decorrelation).

---

## 4. Hyperbox Construction and Overlap Handling

### How a Box Grows (Expansion)

A hyperbox $B_j$ is defined by two corner vectors:
- $V_j$: Min corner coordinates (bottom-left in 2D)
- $W_j$: Max corner coordinates (top-right in 2D)

When a new training point $x$ with class $c$ arrives:
1. Search existing boxes of class $c$.
2. Check if stretching the box to include $x$ exceeds maximum size $\theta$:
   $$\sum_{i=1}^d (\max(w_{ji}, x_i) - \min(v_{ji}, x_i)) \le d \cdot \theta$$
3. If it fits, expand the box:
   $$v_{ji} \leftarrow \min(v_{ji}, x_i), \quad w_{ji} \leftarrow \max(w_{ji}, x_i)$$
4. If it does not fit in any existing box of class $c$, start a new 1-point box at $x$.

### The Centroid

Every time points are placed inside box $B_j$, we store them. The centroid $C_j$ is simply the arithmetic average of all points in that box:

$$C_j = \left( \frac{\sum x_1}{|P_j|}, \frac{\sum x_2}{|P_j|}, \dots, \frac{\sum x_d}{|P_j|} \right)$$

### Overlap Handling Without Contraction

In classic FMNN, if Box A (Class 0) and Box B (Class 1) overlap, the algorithm shrinks them along one axis to create a gap.
In RF-FMNN, **no shrinking occurs**. Both boxes keep their original boundaries.

```
+---------------------------------------+
|  BOX A (Class 0)                      |
|                                       |
|          [Centroid A]                 |
|                   +-------------------+-------------------+
|                   | OVERLAP REGION    |                   |
|                   |        x (Test)   |                   |
|                   |                   |    [Centroid B]   |
+-------------------+-------------------+                   |
                    |                     BOX B (Class 1)   |
                    +---------------------------------------+
```

When a test point $x$ lands inside the overlap zone:
- Distance to Centroid A: $D_A = \|x - C_A\|$
- Distance to Centroid B: $D_B = \|x - C_B\|$
- If $D_A < D_B$, Tree predicts **Class 0**. Otherwise, **Class 1**.

In the upgraded engine, distance is normalized by the width of each box $(w_i - v_i)$ so rectangular boxes are not distorted.

---

## 5. From Training to Testing: The Step-by-Step Data Journey

### Stage 1: Data Preparation
```
Raw Numbers [150.0, 4.2, ...] -> Min-Max Scaler -> [0.85, 0.21, ...]
```
All features are normalized to $[0, 1]$ because fuzzy membership operates on the unit hypercube.

### Stage 2: Training Pipeline
1. Dataset split: 80% Train ($N$ samples), 20% Test.
2. Initialize Ensemble of $M$ trees (e.g. $M=35$).
3. For Tree $m = 1 \dots M$:
   - Draw $N$ instances with replacement from Train.
   - Pick $k = \sqrt{d}$ feature columns.
   - Run hyperbox expansion algorithm on the sampled data.
   - Calculate centroids for all created hyperboxes.
   - Record OOB indices for internal performance tracking.

### Stage 3: Testing / Inference Pipeline
For an unseen test point $x_{test}$:
1. Pass $x_{test}$ to each Tree $m = 1 \dots M$:
   - Tree $m$ isolates its assigned feature columns from $x_{test}$.
   - **Case 1 (Inside 1 Box):** If $x_{test}$ is inside a box of Class $c$, assign Class $c$.
   - **Case 2 (Inside Overlapping Boxes):** If inside boxes of Class 0 and Class 1, calculate distance to each box centroid. Pick the class of the closest centroid.
   - **Case 3 (Outside All Boxes):** Calculate fuzzy membership score $b_j(x)$ to all boxes. The box with highest membership wins.
2. Collect the output from all $M$ trees:
   - Tree 1: Class 0
   - Tree 2: Class 0
   - Tree 3: Class 1
   - ...
3. **Voting**: Count the votes. The class with the highest total wins the final prediction.

---

## 6. Component Architecture Summary Table

| Component | What It Does | Why It Is Needed |
| :--- | :--- | :--- |
| **MinMaxScaler** | Scales raw values into $[0.0, 1.0]$ | FMNN membership equations require unit hypercube inputs |
| **Bootstrap Sampler** | Generates $M$ random row bags with replacement | Prevents the ensemble from overfitting to individual training points |
| **Subspace Selector** | Selects $k=\sqrt{d}$ feature columns randomly | Prevents hyperbox collapse in high-dimensional datasets |
| **Hyperbox Expander** | Grows min ($V$) and max ($W$) bounds up to threshold $\theta$ | Forms interpretable rule boxes representing data clusters |
| **Centroid Tracker** | Computes arithmetic mean of points in each box | Provides an anchor point representing the heart of the cluster |
| **Overlap Resolver** | Compares test sample distance to candidate centroids | Resolves classification ambiguity without destructive contraction |
| **Fuzzy Membership Engine** | Measures distance decay when outside all boxes | Ensures every test point receives a prediction, even in empty space |
| **Ensemble Aggregator** | Combines predictions across all $M$ trees | Delivers low-variance, robust final classification |
