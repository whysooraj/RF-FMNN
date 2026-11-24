import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Hyperbox:
    """Represents a single Fuzzy Min-Max Hyperbox."""
    def __init__(
        self,
        v: np.ndarray,
        w: np.ndarray,
        class_label: int,
        box_id: int,
        sample_sum: np.ndarray = None,
        sample_count: int = 0,
    ):
        self.v = np.array(v, dtype=np.float64)  # Min point
        self.w = np.array(w, dtype=np.float64)  # Max point
        self.class_label = class_label
        self.box_id = box_id
        self.sample_sum = (
            np.zeros_like(self.v, dtype=np.float64)
            if sample_sum is None
            else np.array(sample_sum, dtype=np.float64)
        )
        self.sample_count = int(sample_count)
        self.centroid = np.copy(self.v)

    def add_sample(self, x: np.ndarray):
        self.sample_sum += np.asarray(x, dtype=np.float64)
        self.sample_count += 1

    def update_centroid(self):
        if self.sample_count > 0:
            self.centroid = self.sample_sum / self.sample_count
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

    def get_total_boxes(self) -> int:
        return len(self.hyperboxes)

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

    def _test_and_contract(
        self,
        j_idx: int,
        V_arr: np.ndarray,
        W_arr: np.ndarray,
        labels_arr: np.ndarray,
        n_boxes: int,
    ):
        """
        Simpson 1992 Overlap Test and Contraction between hyperbox j and all boxes of other classes.

        Vectorizes overlap pre-filtering across all opposing boxes simultaneously:
        for valid intervals v <= w, the disjunction of Simpson's 4 overlap conditions
        with delta > 0 is identical to (v_j < w_k) & (v_k < w_j) in all n dimensions.
        Only overlapping opposing boxes run the sequential 4-case contraction.
        """
        v_j = V_arr[j_idx]
        w_j = W_arr[j_idx]
        c_j = labels_arr[j_idx]

        V_all = V_arr[:n_boxes]
        W_all = W_arr[:n_boxes]

        opp_mask = labels_arr[:n_boxes] != c_j
        if not np.any(opp_mask):
            return

        overlap_mask = opp_mask & np.all((v_j < W_all) & (V_all < w_j), axis=1)
        cand_k = np.flatnonzero(overlap_mask)
        if len(cand_k) == 0:
            return

        for k_idx in cand_k:
            v_k = V_arr[k_idx]
            w_k = W_arr[k_idx]

            # Earlier contractions in this loop shrink box j, which can eliminate later overlaps
            if not np.all((v_j < w_k) & (v_k < w_j)):
                continue

            # Across all 4 Simpson overlap cases, delta_i == min(w_j[i] - v_k[i], w_k[i] - v_j[i])
            i = int(np.argmin(np.minimum(w_j - v_k, w_k - v_j)))
            vj, wj = v_j[i], w_j[i]
            vk, wk = v_k[i], w_k[i]

            if vj < vk < wj < wk:
                mid = (wj + vk) / 2.0
                w_j[i] = mid
                v_k[i] = mid
            elif vk < vj < wk < wj:
                mid = (wk + vj) / 2.0
                w_k[i] = mid
                v_j[i] = mid
            elif vj <= vk <= wk <= wj:
                if (wk - vj) < (wj - vk):
                    v_j[i] = wk
                else:
                    w_j[i] = vk
            elif vk <= vj <= wj <= wk:
                if (wk - vj) < (wj - vk):
                    w_k[i] = vj
                else:
                    v_k[i] = wj

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hyperboxes = []

        n_samples = len(X)
        n_features = X.shape[1] if X.ndim == 2 else 0
        label_dtype = self.classes_.dtype if len(self.classes_) else int

        if n_samples == 0:
            self.V_mat = np.empty((0, n_features), dtype=np.float64)
            self.W_mat = np.empty((0, n_features), dtype=np.float64)
            self.labels_mat = np.empty(0, dtype=label_dtype)
            return self

        gamma = self.gamma
        threshold = n_features * self.theta
        denom = 2.0 * n_features

        cap = 32
        n_boxes = 0
        V_arr = np.empty((cap, n_features), dtype=np.float64)
        W_arr = np.empty((cap, n_features), dtype=np.float64)
        labels_arr = np.empty(cap, dtype=label_dtype)
        sum_arr = np.empty((cap, n_features), dtype=np.float64)
        cnt_arr = np.empty(cap, dtype=np.int64)

        for _ in range(self.max_epochs):
            for idx in range(n_samples):
                x = X[idx]
                c = y[idx]

                expand_idx = -1
                if n_boxes > 0:
                    same_idx = np.flatnonzero(labels_arr[:n_boxes] == c)
                    if len(same_idx) > 0:
                        V_c = V_arr[same_idx]
                        W_c = W_arr[same_idx]

                        diff_w = x - W_c
                        diff_v = V_c - x
                        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_w)))
                        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_v)))
                        mems = np.sum(term1 + term2, axis=1) / denom

                        # Stable descending tie order: np.argmax picks earliest box index on ties
                        best_local = int(np.argmax(mems))
                        if float(np.sum(np.maximum(W_c[best_local], x) - np.minimum(V_c[best_local], x))) <= threshold:
                            expand_idx = int(same_idx[best_local])
                        else:
                            eligible = np.sum(np.maximum(W_c, x) - np.minimum(V_c, x), axis=1) <= threshold
                            if np.any(eligible):
                                expand_idx = int(same_idx[int(np.argmax(np.where(eligible, mems, -1.0)))])

                if expand_idx >= 0:
                    np.minimum(V_arr[expand_idx], x, out=V_arr[expand_idx])
                    np.maximum(W_arr[expand_idx], x, out=W_arr[expand_idx])
                    sum_arr[expand_idx] += x
                    cnt_arr[expand_idx] += 1
                    self._test_and_contract(expand_idx, V_arr, W_arr, labels_arr, n_boxes)
                else:
                    if n_boxes == cap:
                        new_cap = cap * 2
                        new_V = np.empty((new_cap, n_features), dtype=np.float64)
                        new_W = np.empty((new_cap, n_features), dtype=np.float64)
                        new_labels = np.empty(new_cap, dtype=label_dtype)
                        new_sum = np.empty((new_cap, n_features), dtype=np.float64)
                        new_cnt = np.empty(new_cap, dtype=np.int64)
                        new_V[:n_boxes] = V_arr
                        new_W[:n_boxes] = W_arr
                        new_labels[:n_boxes] = labels_arr
                        new_sum[:n_boxes] = sum_arr
                        new_cnt[:n_boxes] = cnt_arr
                        V_arr, W_arr, labels_arr, sum_arr, cnt_arr, cap = (
                            new_V, new_W, new_labels, new_sum, new_cnt, new_cap
                        )

                    V_arr[n_boxes] = x
                    W_arr[n_boxes] = x
                    labels_arr[n_boxes] = c
                    sum_arr[n_boxes] = x
                    cnt_arr[n_boxes] = 1
                    j_new = n_boxes
                    n_boxes += 1
                    self._test_and_contract(j_new, V_arr, W_arr, labels_arr, n_boxes)

        self.V_mat = V_arr[:n_boxes].copy()
        self.W_mat = W_arr[:n_boxes].copy()
        self.labels_mat = labels_arr[:n_boxes].copy()
        sums_trim = sum_arr[:n_boxes]
        cnts_trim = cnt_arr[:n_boxes]
        centroids = sums_trim / cnts_trim[:, None]

        hyperboxes = []
        for bid in range(n_boxes):
            box = Hyperbox(
                v=self.V_mat[bid],
                w=self.W_mat[bid],
                class_label=self.labels_mat[bid],
                box_id=bid,
                sample_sum=sums_trim[bid],
                sample_count=int(cnts_trim[bid]),
            )
            box.centroid = centroids[bid].copy()
            hyperboxes.append(box)
        self.hyperboxes = hyperboxes

        return self

    def _membership_matrix(self, X_sub: np.ndarray) -> np.ndarray:
        n_features = self.V_mat.shape[1]
        diff_w = X_sub[:, None, :] - self.W_mat[None, :, :]
        diff_v = self.V_mat[None, :, :] - X_sub[:, None, :]
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        return np.sum(term1 + term2, axis=2) / (2.0 * n_features)

    def predict_sample(self, x: np.ndarray) -> int:
        if not self.hyperboxes:
            return self.classes_[0]
        x = np.asarray(x, dtype=np.float64)
        return self.predict(x.reshape(1, -1))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        out_dtype = self.classes_.dtype if (self.classes_ is not None and len(self.classes_) > 0) else int

        if n_samples == 0:
            return np.empty(0, dtype=out_dtype)
        if not self.hyperboxes:
            return np.full(n_samples, self.classes_[0] if self.classes_ is not None else 0, dtype=out_dtype)

        preds = np.empty(n_samples, dtype=out_dtype)
        n_boxes, n_features = self.V_mat.shape
        batch_size = max(1, min(500, 2_000_000 // max(1, n_boxes * n_features)))

        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            mems = self._membership_matrix(X[start:end])
            preds[start:end] = self.labels_mat[np.argmax(mems, axis=1)]

        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_) if self.classes_ is not None else 0

        if n_samples == 0:
            return np.empty((0, n_classes), dtype=np.float64)
        if not self.hyperboxes:
            return np.full((n_samples, n_classes), 1.0 / n_classes, dtype=np.float64)

        probs = np.empty((n_samples, n_classes), dtype=np.float64)
        n_boxes, n_features = self.V_mat.shape
        batch_size = max(1, min(500, 2_000_000 // max(1, n_boxes * n_features)))
        class_masks = [self.labels_mat == c for c in self.classes_]

        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            mems = self._membership_matrix(X[start:end])
            scores = np.zeros((end - start, n_classes), dtype=np.float64)
            for ci, mask in enumerate(class_masks):
                if np.any(mask):
                    scores[:, ci] = np.max(mems[:, mask], axis=1)
            totals = np.sum(scores, axis=1)
            pos = totals > 0.0
            batch_probs = np.full((end - start, n_classes), 1.0 / n_classes, dtype=np.float64)
            if np.any(pos):
                batch_probs[pos] = scores[pos] / totals[pos, None]
            probs[start:end] = batch_probs

        return probs
