import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

class RawUncontractedHyperbox:
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

    def contains(self, x: np.ndarray, eps: float = 1e-9) -> bool:
        return bool(np.all(x >= self.v - eps) and np.all(x <= self.w + eps))


class RawRFFMNN(BaseEstimator, ClassifierMixin):
    """Original uncontracted FMNN base classifier used in raw proposal."""
    def __init__(self, theta: float = 0.2, gamma: float = 1.0):
        self.theta = theta
        self.gamma = gamma
        self.hyperboxes = []
        self.classes_ = None

    def membership(self, x: np.ndarray, box: RawUncontractedHyperbox) -> float:
        n = len(x)
        diff_w = x - box.w
        diff_v = box.v - x
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        return float(np.sum(term1 + term2) / (2.0 * n))

    def _expansion_criterion(self, x: np.ndarray, box: RawUncontractedHyperbox) -> bool:
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
            candidates = [b for b in self.hyperboxes if b.class_label == c]
            if candidates:
                candidates.sort(key=lambda b: self.membership(x, b), reverse=True)
                for box in candidates:
                    if self._expansion_criterion(x, box):
                        box.v = np.minimum(box.v, x)
                        box.w = np.maximum(box.w, x)
                        box.add_sample(x)
                        expanded = True
                        break

            if not expanded:
                new_box = RawUncontractedHyperbox(v=x, w=x, class_label=c, box_id=box_counter)
                new_box.add_sample(x)
                box_counter += 1
                self.hyperboxes.append(new_box)

        for box in self.hyperboxes:
            box.update_centroid()

        return self

    def predict_sample(self, x: np.ndarray) -> int:
        if not self.hyperboxes:
            return self.classes_[0]

        containing = [b for b in self.hyperboxes if b.contains(x)]
        if containing:
            classes = set(b.class_label for b in containing)
            if len(classes) == 1:
                return next(iter(classes))
            else:
                best_dist = float('inf')
                best_class = None
                for b in containing:
                    dist = float(np.linalg.norm(x - b.centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_class = b.class_label
                return best_class
        else:
            best_mem = -1.0
            best_class = None
            for b in self.hyperboxes:
                mem = self.membership(x, b)
                if mem > best_mem:
                    best_mem = mem
                    best_class = b.class_label
            return best_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x) for x in X])


class RawRFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """Raw un-improved proposal with coupon collector stopping and plurality vote."""
    def __init__(self, theta: float = 0.2, gamma: float = 1.0, max_bags: int = 150, random_state: int = 42):
        self.theta = theta
        self.gamma = gamma
        self.max_bags = max_bags
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = len(X)
        rng = np.random.RandomState(self.random_state)

        covered = set()
        bags = []
        while len(covered) < n_samples and len(bags) < self.max_bags:
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            covered.update(idx)
            bags.append(idx)

        self.models_ = []
        for idx in bags:
            m = RawRFFMNN(theta=self.theta, gamma=self.gamma)
            m.fit(X[idx], y[idx])
            self.models_.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models_])
        final = []
        for i in range(len(X)):
            c = Counter(preds[:, i]).most_common(1)[0][0]
            final.append(c)
        return np.array(final)
