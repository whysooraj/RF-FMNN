import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class UncontractedHyperbox:
    """
    Hyperbox for RF-FMNN.
    Constructed and expanded without overlap contraction.
    Maintains assigned training observations and computes arithmetic centroid
    taking bootstrap sample multiplicity into account.
    """
    def __init__(self, v: np.ndarray, w: np.ndarray, class_label: int, box_id: int):
        self.v = np.array(v, dtype=np.float64)  # Min boundary vector
        self.w = np.array(w, dtype=np.float64)  # Max boundary vector
        self.class_label = class_label
        self.box_id = box_id
        self.samples = []
        self.centroid = np.copy(self.v)

    def add_sample(self, x: np.ndarray):
        """Add training observation. Repetitions are counted according to multiplicity."""
        self.samples.append(np.array(x, dtype=np.float64))

    def update_centroid(self):
        """Centroid calculated as arithmetic mean of all assigned training observations."""
        if len(self.samples) > 0:
            self.centroid = np.mean(self.samples, axis=0)
        else:
            self.centroid = (self.v + self.w) / 2.0

    def contains(self, x: np.ndarray, eps: float = 1e-9) -> bool:
        """Check if test observation falls inside hyperbox boundaries."""
        return bool(np.all(x >= self.v - eps) and np.all(x <= self.w + eps))


class SingleFMNN(BaseEstimator, ClassifierMixin):
    """
    Independent FMNN classifier trained on an individual bootstrap sample.
    - Expands hyperboxes without contraction.
    - Resolves multi-class overlaps using nearest data-based hyperbox centroid.
    - Uses fuzzy membership function when observation is outside all hyperboxes.
    """
    def __init__(self, theta: float = 0.2, gamma: float = 1.0):
        self.theta = theta
        self.gamma = gamma
        self.hyperboxes = []
        self.classes_ = None

    def membership(self, x: np.ndarray, box: UncontractedHyperbox) -> float:
        """
        FMNN membership function (Simpson 1992):
        b_j(x) = (1 / 2n) * sum_{i=1}^n [ max(0, 1 - max(0, gamma * min(1, x_i - w_ji)))
                                       + max(0, 1 - max(0, gamma * min(1, v_ji - x_i))) ]
        """
        n = len(x)
        diff_w = x - box.w
        diff_v = box.v - x
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        return float(np.sum(term1 + term2) / (2.0 * n))

    def _expansion_criterion(self, x: np.ndarray, box: UncontractedHyperbox) -> bool:
        """Check expansion condition against maximum hyperbox size theta."""
        n = len(x)
        expanded_dim = np.maximum(box.w, x) - np.minimum(box.v, x)
        return bool(np.sum(expanded_dim) <= n * self.theta)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hyperboxes = []
        box_counter = 0

        for x, c in zip(X, y):
            expanded = False
            candidate_boxes = [b for b in self.hyperboxes if b.class_label == c]
            if candidate_boxes:
                candidate_boxes.sort(key=lambda b: self.membership(x, b), reverse=True)
                for box in candidate_boxes:
                    if self._expansion_criterion(x, box):
                        box.v = np.minimum(box.v, x)
                        box.w = np.maximum(box.w, x)
                        box.add_sample(x)
                        expanded = True
                        break

            if not expanded:
                new_box = UncontractedHyperbox(v=x, w=x, class_label=c, box_id=box_counter)
                new_box.add_sample(x)
                box_counter += 1
                self.hyperboxes.append(new_box)

        # Calculate centroids for all constructed hyperboxes
        for box in self.hyperboxes:
            box.update_centroid()

        return self

    def predict_sample(self, x: np.ndarray) -> int:
        if not self.hyperboxes:
            return self.classes_[0]

        # Step 7: Check containing hyperboxes
        containing_boxes = [b for b in self.hyperboxes if b.contains(x)]

        if containing_boxes:
            contained_classes = set(b.class_label for b in containing_boxes)
            if len(contained_classes) == 1:
                return next(iter(contained_classes))
            else:
                # Overlap between different classes: resolve using nearest data-based centroid
                best_dist = float('inf')
                best_class = None
                for b in containing_boxes:
                    dist = float(np.linalg.norm(x - b.centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_class = b.class_label
                return best_class
        else:
            # Step 8: Outside all hyperboxes -> use FMNN membership function
            best_membership = -1.0
            best_class = None
            for box in self.hyperboxes:
                mem = self.membership(x, box)
                if mem > best_membership:
                    best_membership = mem
                    best_class = box.class_label
            return best_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self.predict_sample(x) for x in X])


class RFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN).
    
    Combines multiple independent FMNN models using bootstrap random sampling
    and ensemble majority/plurality voting.
    
    Bootstrap samples are generated until the union of unique observation IDs
    covers the complete training dataset.
    Hyperboxes are constructed without contraction; overlapping regions
    are resolved via nearest data-based centroids.
    """
    def __init__(self, theta: float = 0.2, gamma: float = 1.0, max_bags_limit: int = 150, random_state: int = 42):
        self.theta = theta
        self.gamma = gamma
        self.max_bags_limit = max_bags_limit
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None
        self.n_bags_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = len(X)
        rng = np.random.RandomState(self.random_state)

        # Step 4: Bootstrap random sampling with replacement until complete coverage
        covered_ids = set()
        bootstrap_bags = []

        while len(covered_ids) < n_samples and len(bootstrap_bags) < self.max_bags_limit:
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            covered_ids.update(boot_idx)
            bootstrap_bags.append(boot_idx)

        self.n_bags_ = len(bootstrap_bags)
        self.models_ = []

        # Step 5 & 6: Train independent FMNN models on each bootstrap sample
        for boot_idx in bootstrap_bags:
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            model = SingleFMNN(theta=self.theta, gamma=self.gamma)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Step 9 & 10: Collect independent predictions and perform majority voting."""
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.array([model.predict(X) for model in self.models_])
        final_preds = []
        for i in range(len(X)):
            sample_preds = all_preds[:, i]
            counts = Counter(sample_preds)
            majority_class = counts.most_common(1)[0][0]
            final_preds.append(majority_class)
        return np.array(final_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        all_preds = np.array([model.predict(X) for model in self.models_])
        probs = []
        for i in range(len(X)):
            sample_preds = all_preds[:, i]
            counts = Counter(sample_preds)
            prob_vec = [counts.get(c, 0) / self.n_bags_ for c in self.classes_]
            probs.append(prob_vec)
        return np.array(probs)
