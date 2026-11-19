import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnhancedHyperbox:
    """Hyperbox with feature subspace projection, sample support, and normalized centroid."""
    def __init__(self, v: np.ndarray, w: np.ndarray, class_label: int, box_id: int):
        self.v = np.array(v, dtype=np.float64)
        self.w = np.array(w, dtype=np.float64)
        self.class_label = class_label
        self.box_id = box_id
        self.samples = []
        self.centroid = np.copy(self.v)

    def add_sample(self, x: np.ndarray):
        self.samples.append(np.array(x, dtype=np.float64))

    def update_centroid(self):
        if len(self.samples) > 0:
            self.centroid = np.mean(self.samples, axis=0)
        else:
            self.centroid = (self.v + self.w) / 2.0

    @property
    def support(self) -> int:
        return len(self.samples)

    def contains(self, x: np.ndarray, eps: float = 1e-9) -> bool:
        return bool(np.all(x >= self.v - eps) and np.all(x <= self.w + eps))

    def normalized_distance_to_centroid(self, x: np.ndarray, eps: float = 1e-5) -> float:
        """
        Dimension-normalized Mahalanobis/ellipsoidal distance:
        Accounts for hyperbox aspect ratio across each feature.
        """
        span = (self.w - self.v) + eps
        diff = (x - self.centroid) / span
        return float(np.sqrt(np.sum(diff ** 2)))


class SubspaceFMNN(BaseEstimator, ClassifierMixin):
    """
    Subspace-projected Fuzzy Min-Max base estimator.
    Trained on a random feature subspace with adaptive theta and soft membership output.
    """
    def __init__(self, feature_indices: np.ndarray, theta: float = 0.2, gamma: float = 1.0, beta: float = 1.0):
        self.feature_indices = feature_indices
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
        self.hyperboxes = []
        self.classes_ = None

    def membership(self, x_sub: np.ndarray, box: EnhancedHyperbox) -> float:
        n = len(x_sub)
        diff_w = x_sub - box.w
        diff_v = box.v - x_sub
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        return float(np.sum(term1 + term2) / (2.0 * n))

    def _expansion_criterion(self, x_sub: np.ndarray, box: EnhancedHyperbox) -> bool:
        n = len(x_sub)
        expanded_dim = np.maximum(box.w, x_sub) - np.minimum(box.v, x_sub)
        return bool(np.sum(expanded_dim) <= n * self.theta)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_sub = np.asarray(X[:, self.feature_indices], dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hyperboxes = []
        box_counter = 0

        for idx, (x_sub, c) in enumerate(zip(X_sub, y)):
            expanded = False
            candidate_boxes = [b for b in self.hyperboxes if b.class_label == c]
            if candidate_boxes:
                candidate_boxes.sort(key=lambda b: self.membership(x_sub, b), reverse=True)
                for box in candidate_boxes:
                    if self._expansion_criterion(x_sub, box):
                        box.v = np.minimum(box.v, x_sub)
                        box.w = np.maximum(box.w, x_sub)
                        box.add_sample(x_sub)
                        expanded = True
                        break

            if not expanded:
                new_box = EnhancedHyperbox(v=x_sub, w=x_sub, class_label=c, box_id=box_counter)
                new_box.add_sample(x_sub)
                box_counter += 1
                self.hyperboxes.append(new_box)

        for box in self.hyperboxes:
            box.update_centroid()

        return self

    def predict_fuzzy_scores(self, x_full: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
        """
        Returns continuous class score vector for x using normalized distance and fuzzy membership.
        """
        x_sub = x_full[self.feature_indices]
        scores = {c: 0.0 for c in all_classes}

        containing_boxes = [b for b in self.hyperboxes if b.contains(x_sub)]

        if containing_boxes:
            # Overlap handling with normalized ellipsoidal distance and support weighting
            for b in containing_boxes:
                norm_dist = b.normalized_distance_to_centroid(x_sub)
                support_weight = np.tanh(0.5 * b.support)
                # Proximity score in [0, 1]
                proximity = support_weight * np.exp(-self.beta * norm_dist)
                if proximity > scores[b.class_label]:
                    scores[b.class_label] = proximity
        else:
            # Outside containing hyperboxes: standard fuzzy membership with support scaling
            for b in self.hyperboxes:
                mem = self.membership(x_sub, b)
                support_weight = np.tanh(0.5 * b.support)
                weighted_mem = support_weight * mem
                if weighted_mem > scores[b.class_label]:
                    scores[b.class_label] = weighted_mem

        vec = np.array([scores[c] for c in all_classes], dtype=np.float64)
        total = np.sum(vec)
        if total > 0:
            return vec / total
        return np.ones(len(all_classes), dtype=np.float64) / len(all_classes)


class EnhancedRFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Enhanced Random Forest Fuzzy Min-Max Neural Network (ERF-FMNN).
    Senior ML Researcher Design:
    1. Random Subspace Feature Selection (decorrelated base learners).
    2. Multi-Scale Granularity (Stochastic Theta Diversification).
    3. Dimension-Normalized Ellipsoidal Metric for Overlap.
    4. Support-Weighted Outlier Regularization.
    5. Soft Continuous Fuzzy Consensus Voting.
    """
    def __init__(
        self,
        n_estimators: int = 30,
        theta_range: tuple = (0.1, 0.4),
        max_features: str = 'sqrt',
        gamma: float = 1.0,
        beta: float = 1.0,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.theta_range = theta_range
        self.max_features = max_features
        self.gamma = gamma
        self.beta = beta
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Determine subspace dimensionality
        if self.max_features == 'sqrt':
            k_features = max(2, int(np.round(np.sqrt(n_features))))
        elif self.max_features == 'log2':
            k_features = max(2, int(np.round(np.log2(n_features))))
        elif isinstance(self.max_features, float):
            k_features = max(2, int(np.round(self.max_features * n_features)))
        else:
            k_features = n_features
        k_features = min(k_features, n_features)

        self.models_ = []

        for m in range(self.n_estimators):
            # 1. Bootstrap sample rows
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # 2. Random feature subspace
            feat_idx = rng.choice(n_features, size=k_features, replace=False)

            # 3. Adaptive theta multi-scale granulation
            theta_m = rng.uniform(self.theta_range[0], self.theta_range[1])

            model = SubspaceFMNN(
                feature_indices=feat_idx,
                theta=theta_m,
                gamma=self.gamma,
                beta=self.beta
            )
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)
        aggregated_scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for model in self.models_:
            for i, x in enumerate(X):
                scores = model.predict_fuzzy_scores(x, self.classes_)
                aggregated_scores[i] += scores

        # Normalize across classes
        row_sums = aggregated_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aggregated_scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]
