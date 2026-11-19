import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class UnifiedHyperbox:
    """
    Robust Hyperbox featuring:
    - Arithmetic data centroid with multiplicity tracking
    - Dimension-normalized Mahalanobis/ellipsoidal metric
    - Support regularization: tanh(0.5 * |P_j|)
    - Continuous Centroid-Anchor Fuzzy Membership
    """
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

    def normalized_distance(self, x: np.ndarray, eps: float = 1e-5) -> float:
        """Dimension-normalized distance scaling by axis-specific span."""
        span = (self.w - self.v) + eps
        diff = (x - self.centroid) / span
        return float(np.sqrt(np.sum(diff ** 2)))

    def continuous_membership(self, x: np.ndarray, gamma: float = 1.0, beta: float = 0.5) -> float:
        """
        Unified Centroid-Anchor Fuzzy Membership:
        mu_j(x) = tanh(0.5 * |P_j|) * b_j(x) * exp(-beta * D_norm(x, C_j, B_j))
        Smooth and continuous across entire input space.
        """
        n = len(x)
        diff_w = x - self.w
        diff_v = self.v - x
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_v)))
        b_edge = float(np.sum(term1 + term2) / (2.0 * n))

        d_norm = self.normalized_distance(x)
        w_support = float(np.tanh(0.5 * max(1, self.support)))

        return float(w_support * b_edge * np.exp(-beta * d_norm))


class BaseUncontractedFMNN(BaseEstimator, ClassifierMixin):
    """
    Subspace uncontracted FMNN base classifier.
    Constructs hyperboxes without contraction and evaluates continuous fuzzy membership.
    """
    def __init__(self, feature_indices: np.ndarray = None, theta: float = 0.2, gamma: float = 1.0, beta: float = 0.5):
        self.feature_indices = feature_indices
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
        self.hyperboxes = []
        self.classes_ = None

    def _expansion_criterion(self, x: np.ndarray, box: UnifiedHyperbox) -> bool:
        n = len(x)
        expanded_dim = np.maximum(box.w, x) - np.minimum(box.v, x)
        return bool(np.sum(expanded_dim) <= n * self.theta)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.feature_indices is not None:
            X_feat = X[:, self.feature_indices]
        else:
            X_feat = X

        X_feat = np.asarray(X_feat, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hyperboxes = []
        box_counter = 0

        for x, c in zip(X_feat, y):
            expanded = False
            candidates = [b for b in self.hyperboxes if b.class_label == c]
            if candidates:
                # Sort candidates by membership
                candidates.sort(key=lambda b: b.continuous_membership(x, self.gamma, self.beta), reverse=True)
                for box in candidates:
                    if self._expansion_criterion(x, box):
                        box.v = np.minimum(box.v, x)
                        box.w = np.maximum(box.w, x)
                        box.add_sample(x)
                        expanded = True
                        break

            if not expanded:
                new_box = UnifiedHyperbox(v=x, w=x, class_label=c, box_id=box_counter)
                new_box.add_sample(x)
                box_counter += 1
                self.hyperboxes.append(new_box)

        for box in self.hyperboxes:
            box.update_centroid()

        return self

    def compute_class_memberships(self, x_full: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
        if self.feature_indices is not None:
            x = x_full[self.feature_indices]
        else:
            x = x_full

        class_scores = {c: 0.0 for c in all_classes}
        for box in self.hyperboxes:
            mem = box.continuous_membership(x, self.gamma, self.beta)
            if mem > class_scores[box.class_label]:
                class_scores[box.class_label] = mem

        return np.array([class_scores[c] for c in all_classes], dtype=np.float64)


class RFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Upgraded Production RF-FMNN Architecture (Resolving all 5 theoretical flaws).
    
    1. Fixed M in [30, 50] bags with OOB evaluation (eliminates coupon collector flaw).
    2. Random Subspace feature bagging k = round(sqrt(d)) (solves high-dimensional box collapse).
    3. Dimension-normalized ellipsoidal distance D_norm (solves rectangular aspect ratio bias).
    4. Support regularization tanh(0.5 * |P_j|) (suppresses outlier singletons).
    5. Continuous Centroid-Anchor Fuzzy Membership (eliminates boundary discontinuity).
    6. Soft Fuzzy Consensus Voting (preserves continuous confidence margins).
    """
    def __init__(
        self,
        n_estimators: int = 35,
        theta_range: tuple = (0.15, 0.35),
        max_features: str = 'sqrt',
        gamma: float = 1.0,
        beta: float = 0.5,
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.theta_range = theta_range
        self.max_features = max_features
        self.gamma = gamma
        self.beta = beta
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.models_ = []
        self.classes_ = None
        self.oob_score_ = None
        self.oob_indices_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Feature subspace dimensionality
        if self.max_features == 'sqrt':
            k_feat = max(2, int(np.round(np.sqrt(n_features))))
        elif self.max_features == 'log2':
            k_feat = max(2, int(np.round(np.log2(n_features))))
        elif isinstance(self.max_features, float):
            k_feat = max(2, int(np.round(self.max_features * n_features)))
        else:
            k_feat = n_features
        k_feat = min(k_feat, n_features)

        self.models_ = []
        self.oob_indices_ = []
        oob_predictions = {i: [] for i in range(n_samples)}

        for m in range(self.n_estimators):
            # 1. Bootstrap instance sampling
            if self.bootstrap:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                oob_idx = list(set(range(n_samples)) - set(boot_idx))
            else:
                boot_idx = np.arange(n_samples)
                oob_idx = []
            self.oob_indices_.append(oob_idx)

            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            # 2. Random feature subspace selection
            feat_idx = rng.choice(n_features, size=k_feat, replace=False)

            # 3. Multi-scale stochastic expansion threshold theta
            theta_m = rng.uniform(self.theta_range[0], self.theta_range[1])

            model = BaseUncontractedFMNN(
                feature_indices=feat_idx,
                theta=theta_m,
                gamma=self.gamma,
                beta=self.beta
            )
            model.fit(X_boot, y_boot)
            self.models_.append(model)

            # Accumulate OOB predictions
            for idx in oob_idx:
                scores = model.compute_class_memberships(X[idx], self.classes_)
                oob_predictions[idx].append(scores)

        # Calculate Out-of-Bag (OOB) accuracy if enabled
        if self.oob_score and self.bootstrap:
            correct = 0
            evaluated = 0
            for idx in range(n_samples):
                preds = oob_predictions[idx]
                if len(preds) > 0:
                    avg_score = np.mean(preds, axis=0)
                    pred_class = self.classes_[np.argmax(avg_score)]
                    if pred_class == y[idx]:
                        correct += 1
                    evaluated += 1
            self.oob_score_ = float(correct / evaluated) if evaluated > 0 else 0.0

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft Fuzzy Evidence Consensus."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)
        aggregated_scores = np.zeros((n_samples, n_classes), dtype=np.float64)

        for model in self.models_:
            for i, x in enumerate(X):
                scores = model.compute_class_memberships(x, self.classes_)
                aggregated_scores[i] += scores

        row_sums = aggregated_scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aggregated_scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]
