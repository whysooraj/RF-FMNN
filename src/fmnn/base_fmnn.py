import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Hyperbox:
    """Represents a single Fuzzy Min-Max Hyperbox."""
    def __init__(self, v: np.ndarray, w: np.ndarray, class_label: int, box_id: int):
        self.v = np.array(v, dtype=np.float64)  # Min point
        self.w = np.array(w, dtype=np.float64)  # Max point
        self.class_label = class_label
        self.box_id = box_id
        self.sample_indices = []
        self.samples = []
        self.centroid = np.copy(self.v)

    def update_centroid(self):
        if len(self.samples) > 0:
            self.centroid = np.mean(self.samples, axis=0)
        else:
            self.centroid = (self.v + self.w) / 2.0

    def contains(self, x: np.ndarray) -> bool:
        """Check if vector x is completely inside hyperbox."""
        return bool(np.all(x >= self.v - 1e-9) and np.all(x <= self.w + 1e-9))

class BaseFMNN(BaseEstimator, ClassifierMixin):
    """
    Standard Simpson (1992) Fuzzy Min-Max Neural Network Classifier.
    Includes Expansion, Overlap Testing, and Contraction.
    """
    def __init__(self, theta: float = 0.2, gamma: float = 1.0, max_epochs: int = 1):
        self.theta = theta
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.hyperboxes = []
        self.classes_ = None

    def membership(self, x: np.ndarray, box: Hyperbox) -> float:
        """
        Simpson 1992 fuzzy hyperbox membership function:
        b_j(x) = (1 / 2n) * sum_{i=1}^n [ max(0, 1 - max(0, gamma * min(1, x_i - w_ji)))
                                       + max(0, 1 - max(0, gamma * min(1, v_ji - x_i))) ]
        """
        n = len(x)
        diff_w = x - box.w
        diff_v = box.v - x
        
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        
        return float(np.sum(term1 + term2) / (2.0 * n))

    def _expansion_criterion(self, x: np.ndarray, box: Hyperbox) -> bool:
        """Check if expanding box to include x satisfies max size theta."""
        n = len(x)
        expanded_dim = np.maximum(box.w, x) - np.minimum(box.v, x)
        return bool(np.sum(expanded_dim) <= n * self.theta)

    def _test_and_contract(self, j_idx: int):
        """
        Simpson 1992 Overlap Test and Contraction between hyperbox j and all boxes of other classes.
        """
        box_j = self.hyperboxes[j_idx]
        n_dim = len(box_j.v)

        for k_idx, box_k in enumerate(self.hyperboxes):
            if box_k.class_label == box_j.class_label:
                continue

            delta_min = float('inf')
            dim_min = -1
            case_min = -1

            overlap_in_all_dims = True
            for i in range(n_dim):
                v_j, w_j = box_j.v[i], box_j.w[i]
                v_k, w_k = box_k.v[i], box_k.w[i]

                delta = -1.0
                case = -1

                if v_j < v_k < w_j < w_k:
                    case = 1
                    delta = w_j - v_k
                elif v_k < v_j < w_k < w_j:
                    case = 2
                    delta = w_k - v_j
                elif v_j <= v_k <= w_k <= w_j:
                    case = 3
                    delta = min(w_j - v_k, w_k - v_j)
                elif v_k <= v_j <= w_j <= w_k:
                    case = 4
                    delta = min(w_k - v_j, w_j - v_k)

                if delta > 0:
                    if delta < delta_min:
                        delta_min = delta
                        dim_min = i
                        case_min = case
                else:
                    overlap_in_all_dims = False
                    break

            if overlap_in_all_dims and dim_min != -1:
                # Contraction along dimension dim_min
                i = dim_min
                v_j, w_j = box_j.v[i], box_j.w[i]
                v_k, w_k = box_k.v[i], box_k.w[i]

                if case_min == 1:
                    mid = (w_j + v_k) / 2.0
                    box_j.w[i] = mid
                    box_k.v[i] = mid
                elif case_min == 2:
                    mid = (w_k + v_j) / 2.0
                    box_k.w[i] = mid
                    box_j.v[i] = mid
                elif case_min == 3:
                    if (w_k - v_j) < (w_j - v_k):
                        box_j.v[i] = box_k.w[i]
                    else:
                        box_j.w[i] = box_k.v[i]
                elif case_min == 4:
                    if (w_k - v_j) < (w_j - v_k):
                        box_k.w[i] = box_j.v[i]
                    else:
                        box_k.v[i] = box_j.w[i]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hyperboxes = []
        box_counter = 0

        for epoch in range(self.max_epochs):
            for idx, (x, c) in enumerate(zip(X, y)):
                expanded = False
                
                # Find matching class boxes sorted by membership
                candidate_boxes = [
                    (i, b) for i, b in enumerate(self.hyperboxes) if b.class_label == c
                ]
                if candidate_boxes:
                    candidate_boxes.sort(key=lambda t: self.membership(x, t[1]), reverse=True)
                    for b_idx, box in candidate_boxes:
                        if self._expansion_criterion(x, box):
                            box.v = np.minimum(box.v, x)
                            box.w = np.maximum(box.w, x)
                            box.sample_indices.append(idx)
                            box.samples.append(x)
                            self._test_and_contract(b_idx)
                            expanded = True
                            break

                if not expanded:
                    # Create new hyperbox
                    new_box = Hyperbox(v=x, w=x, class_label=c, box_id=box_counter)
                    new_box.sample_indices.append(idx)
                    new_box.samples.append(x)
                    box_counter += 1
                    self.hyperboxes.append(new_box)
                    self._test_and_contract(len(self.hyperboxes) - 1)

        # Update centroids
        for box in self.hyperboxes:
            box.update_centroid()

        return self

    def predict_sample(self, x: np.ndarray) -> int:
        if not self.hyperboxes:
            return self.classes_[0]

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        probs = []
        for x in X:
            class_scores = {c: 0.0 for c in self.classes_}
            for box in self.hyperboxes:
                mem = self.membership(x, box)
                if mem > class_scores[box.class_label]:
                    class_scores[box.class_label] = mem
            total = sum(class_scores.values())
            if total > 0:
                p = [class_scores[c] / total for c in self.classes_]
            else:
                p = [1.0 / len(self.classes_) for _ in self.classes_]
            probs.append(p)
        return np.array(probs)
