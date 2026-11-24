import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _resolve_max_features(max_features, n_features: int) -> int:
    """Resolve max_features specification into an integer subspace size k in [1, n_features]."""
    if max_features is None:
        return n_features
    if max_features == 'sqrt':
        return max(1, int(np.floor(np.sqrt(n_features))))
    if max_features == 'log2':
        return max(1, int(np.floor(np.log2(n_features))))
    if isinstance(max_features, (int, np.integer)) and not isinstance(max_features, bool):
        if max_features < 1 or max_features > n_features:
            raise ValueError(f"max_features={max_features} must be in [1, {n_features}]")
        return int(max_features)
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"float max_features={max_features} must be in (0.0, 1.0]")
        return max(1, int(np.floor(max_features * n_features)))
    raise ValueError(f"Unsupported max_features: {max_features!r}")


def _generate_bag_specs(
    n_samples: int,
    n_features: int,
    n_estimators: int,
    max_features,
    max_bags_limit: int,
    random_state: int,
):
    """
    Generate deterministic (boot_idx, feat_idx) specifications for RF-FMNN bags.
    Uses n_estimators bags when provided; otherwise draws bootstrap samples
    until 100% row coverage or max_bags_limit (coupon-collector rule).
    """
    rng = np.random.RandomState(random_state)
    k = _resolve_max_features(max_features, n_features)
    use_subspace = k < n_features

    bag_specs = []
    if n_estimators is not None:
        for _ in range(n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            feat_idx = np.sort(rng.choice(n_features, size=k, replace=False)) if use_subspace else None
            bag_specs.append((boot_idx, feat_idx))
    else:
        covered = np.zeros(n_samples, dtype=bool)
        covered_count = 0
        while covered_count < n_samples and len(bag_specs) < max_bags_limit:
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            covered[boot_idx] = True
            covered_count = int(covered.sum())
            feat_idx = np.sort(rng.choice(n_features, size=k, replace=False)) if use_subspace else None
            bag_specs.append((boot_idx, feat_idx))

    return bag_specs


class UncontractedHyperbox:
    """
    Hyperbox for RF-FMNN.
    Constructed and expanded without overlap contraction.
    Tracks running sum and sample count in O(d) memory to compute exact
    arithmetic centroids taking bootstrap sample multiplicity into account.
    """
    def __init__(
        self,
        v: np.ndarray,
        w: np.ndarray,
        class_label: int,
        box_id: int,
        sample_sum: np.ndarray = None,
        sample_count: int = 0,
    ):
        self.v = np.array(v, dtype=np.float64)  # Min boundary vector
        self.w = np.array(w, dtype=np.float64)  # Max boundary vector
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
        """Add training observation. Repetitions are counted according to multiplicity."""
        self.sample_sum += np.asarray(x, dtype=np.float64)
        self.sample_count += 1

    def update_centroid(self):
        """Centroid calculated as arithmetic mean of all assigned training observations."""
        if self.sample_count > 0:
            self.centroid = self.sample_sum / self.sample_count
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
    def __init__(self, theta: float = 0.2, gamma: float = 1.0, prune: bool = False):
        self.theta = theta
        self.gamma = gamma
        self.prune = prune
        self.hyperboxes = []
        self.classes_ = None

    def get_total_boxes(self) -> int:
        return len(self.hyperboxes)

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

        n_samples = len(X)
        n_features = X.shape[1] if X.ndim == 2 else 0
        if n_samples == 0:
            self.V_mat = np.empty((0, n_features), dtype=np.float64)
            self.W_mat = np.empty((0, n_features), dtype=np.float64)
            self.centroids_mat = np.empty((0, n_features), dtype=np.float64)
            self.labels_mat = np.empty(0, dtype=self.classes_.dtype if len(self.classes_) else int)
            return self

        gamma = self.gamma
        threshold = n_features * self.theta
        init_cap = 16
        box_counter = 0

        # Per-class dynamic arrays: [V_arr, W_arr, sum_arr, cnt_arr, id_arr, count, cap]
        class_state = {}

        for i in range(n_samples):
            x = X[i]
            c = y[i]
            state = class_state.get(c)

            if state is None:
                V_arr = np.empty((init_cap, n_features), dtype=np.float64)
                W_arr = np.empty((init_cap, n_features), dtype=np.float64)
                sum_arr = np.empty((init_cap, n_features), dtype=np.float64)
                cnt_arr = np.empty(init_cap, dtype=np.int64)
                id_arr = np.empty(init_cap, dtype=np.int64)
                V_arr[0] = x
                W_arr[0] = x
                sum_arr[0] = x
                cnt_arr[0] = 1
                id_arr[0] = box_counter
                box_counter += 1
                class_state[c] = [V_arr, W_arr, sum_arr, cnt_arr, id_arr, 1, init_cap]
                continue

            V_arr, W_arr, sum_arr, cnt_arr, id_arr, k, cap = state
            V_c = V_arr[:k]
            W_c = W_arr[:k]

            # Vectorized Simpson membership across all k boxes of class c (omit 1/(2d) constant)
            diff_w = x - W_c
            diff_v = V_c - x
            term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_w)))
            term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, gamma * np.minimum(1.0, diff_v)))
            mems = np.sum(term1 + term2, axis=1)

            # Fast path: check highest-membership box first; otherwise vectorized eligibility scan
            best = int(np.argmax(mems))
            if float(np.sum(np.maximum(W_c[best], x) - np.minimum(V_c[best], x))) <= threshold:
                expand_idx = best
            else:
                eligible = np.sum(np.maximum(W_c, x) - np.minimum(V_c, x), axis=1) <= threshold
                if np.any(eligible):
                    # Exact equivalent to scanning np.argsort(-mems, kind='stable') for first eligible
                    expand_idx = int(np.argmax(np.where(eligible, mems, -1.0)))
                else:
                    expand_idx = -1

            if expand_idx >= 0:
                np.minimum(V_arr[expand_idx], x, out=V_arr[expand_idx])
                np.maximum(W_arr[expand_idx], x, out=W_arr[expand_idx])
                sum_arr[expand_idx] += x
                cnt_arr[expand_idx] += 1
            else:
                if k == cap:
                    new_cap = cap * 2
                    new_V = np.empty((new_cap, n_features), dtype=np.float64)
                    new_W = np.empty((new_cap, n_features), dtype=np.float64)
                    new_sum = np.empty((new_cap, n_features), dtype=np.float64)
                    new_cnt = np.empty(new_cap, dtype=np.int64)
                    new_id = np.empty(new_cap, dtype=np.int64)
                    new_V[:k] = V_arr
                    new_W[:k] = W_arr
                    new_sum[:k] = sum_arr
                    new_cnt[:k] = cnt_arr
                    new_id[:k] = id_arr
                    V_arr, W_arr, sum_arr, cnt_arr, id_arr, cap = (
                        new_V, new_W, new_sum, new_cnt, new_id, new_cap
                    )
                    state[0], state[1], state[2], state[3], state[4], state[6] = (
                        V_arr, W_arr, sum_arr, cnt_arr, id_arr, cap
                    )
                V_arr[k] = x
                W_arr[k] = x
                sum_arr[k] = x
                cnt_arr[k] = 1
                id_arr[k] = box_counter
                box_counter += 1
                state[5] = k + 1

        # Build UncontractedHyperbox list (no raw samples list) and vectorized prediction matrices
        total_boxes = box_counter
        self.V_mat = np.empty((total_boxes, n_features), dtype=np.float64)
        self.W_mat = np.empty((total_boxes, n_features), dtype=np.float64)
        self.centroids_mat = np.empty((total_boxes, n_features), dtype=np.float64)
        self.labels_mat = np.empty(total_boxes, dtype=self.classes_.dtype)
        hyperboxes = [None] * total_boxes

        for c, state in class_state.items():
            k = state[5]
            V_c = state[0][:k]
            W_c = state[1][:k]
            sum_c = state[2][:k]
            cnt_c = state[3][:k]
            ids_c = state[4][:k]
            centroids_c = sum_c / cnt_c[:, None]

            self.V_mat[ids_c] = V_c
            self.W_mat[ids_c] = W_c
            self.centroids_mat[ids_c] = centroids_c
            self.labels_mat[ids_c] = c

            for j in range(k):
                bid = int(ids_c[j])
                box = UncontractedHyperbox(
                    v=V_c[j],
                    w=W_c[j],
                    class_label=c,
                    box_id=bid,
                    sample_sum=sum_c[j],
                    sample_count=int(cnt_c[j]),
                )
                box.centroid = centroids_c[j].copy()
                hyperboxes[bid] = box

        self.hyperboxes = hyperboxes
        if self.prune:
            self._consolidate_boxes()
        return self

    def _consolidate_boxes(self, prune_scale: float = 1.25):
        """
        Post-training same-class hyperbox consolidation pass.
        Because uncontracted online expansion is monotonic, no two distinct
        same-class boxes can satisfy union size <= d * theta at 1.0x scale;
        relaxing the consolidation bound by prune_scale (default 1.25x theta)
        greedily merges adjacent same-class hyperbox fragments while combining
        sample_sum and sample_count for exact data-based centroids.
        """
        n_boxes = len(self.hyperboxes)
        if n_boxes <= 1:
            return

        n_features = self.V_mat.shape[1]
        threshold = n_features * min(1.0, self.theta * prune_scale)
        sum_mat = np.array([b.sample_sum for b in self.hyperboxes], dtype=np.float64)
        cnt_mat = np.array([b.sample_count for b in self.hyperboxes], dtype=np.int64)

        kept_records = []
        for c in self.classes_:
            idx = np.flatnonzero(self.labels_mat == c)
            k = len(idx)
            if k == 0:
                continue
            V_c = self.V_mat[idx].copy()
            W_c = self.W_mat[idx].copy()
            sum_c = sum_mat[idx].copy()
            cnt_c = cnt_mat[idx].copy()

            if k > 1:
                active = np.ones(k, dtype=bool)
                for a in range(k):
                    if not active[a]:
                        continue
                    while True:
                        rem = np.flatnonzero(active[a + 1:]) + (a + 1)
                        if len(rem) == 0:
                            break
                        u_v = np.minimum(V_c[a], V_c[rem])
                        u_w = np.maximum(W_c[a], W_c[rem])
                        sizes = np.sum(u_w - u_v, axis=1)
                        best_local = int(np.argmin(sizes))
                        if sizes[best_local] <= threshold:
                            b = rem[best_local]
                            V_c[a] = u_v[best_local]
                            W_c[a] = u_w[best_local]
                            sum_c[a] += sum_c[b]
                            cnt_c[a] += cnt_c[b]
                            active[b] = False
                        else:
                            break
                surv = np.flatnonzero(active)
            else:
                surv = np.array([0], dtype=np.int64)

            for pos in surv:
                kept_records.append((int(idx[pos]), V_c[pos], W_c[pos], sum_c[pos], int(cnt_c[pos]), c))

        # Preserve deterministic original box creation order
        kept_records.sort(key=lambda r: r[0])
        m = len(kept_records)

        self.V_mat = np.empty((m, n_features), dtype=np.float64)
        self.W_mat = np.empty((m, n_features), dtype=np.float64)
        self.centroids_mat = np.empty((m, n_features), dtype=np.float64)
        self.labels_mat = np.empty(m, dtype=self.classes_.dtype)
        hyperboxes = []

        for bid, (_, v, w, s_sum, s_cnt, c) in enumerate(kept_records):
            centroid = (s_sum / s_cnt) if s_cnt > 0 else ((v + w) / 2.0)
            self.V_mat[bid] = v
            self.W_mat[bid] = w
            self.centroids_mat[bid] = centroid
            self.labels_mat[bid] = c
            box = UncontractedHyperbox(
                v=v,
                w=w,
                class_label=c,
                box_id=bid,
                sample_sum=s_sum,
                sample_count=s_cnt,
            )
            box.centroid = centroid.copy()
            hyperboxes.append(box)

        self.hyperboxes = hyperboxes

    def predict_sample(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64)
        if not self.hyperboxes:
            return self.classes_[0]
        return self.predict(x.reshape(1, -1))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch prediction with containment, centroid resolution, and deterministic tie-breaking."""
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
        eps = 1e-9

        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            X_sub = X[start:end]

            # Vectorized containment matrix: shape (batch_size, n_boxes)
            in_box = np.all(
                (X_sub[:, None, :] >= self.V_mat[None, :, :] - eps)
                & (X_sub[:, None, :] <= self.W_mat[None, :, :] + eps),
                axis=2,
            )

            for i in range(len(X_sub)):
                idx = start + i
                cand = np.where(in_box[i])[0]

                if len(cand) > 0:
                    cand_labels = self.labels_mat[cand]
                    first_lbl = cand_labels[0]
                    if np.all(cand_labels == first_lbl):
                        preds[idx] = first_lbl
                    else:
                        # Squared Euclidean distance to candidate centroids;
                        # deterministic tie-break by smallest class label
                        dists = np.sum((self.centroids_mat[cand] - X_sub[i]) ** 2, axis=1)
                        best_pos = int(np.lexsort((cand_labels, dists))[0])
                        preds[idx] = cand_labels[best_pos]
                else:
                    # Fallback to Simpson fuzzy membership (vectorized over all boxes)
                    diff_w = X_sub[i] - self.W_mat
                    diff_v = self.V_mat - X_sub[i]
                    term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
                    term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
                    mems = np.sum(term1 + term2, axis=1)
                    max_mem = np.max(mems)
                    tied = np.where(mems >= max_mem - 1e-12)[0]
                    if len(tied) == 1:
                        preds[idx] = self.labels_mat[tied[0]]
                    else:
                        # Handle ties (including out-of-range saturation) via nearest centroid, then label
                        dists = np.sum((self.centroids_mat[tied] - X_sub[i]) ** 2, axis=1)
                        tied_labels = self.labels_mat[tied]
                        best_pos = int(np.lexsort((tied_labels, dists))[0])
                        preds[idx] = tied_labels[best_pos]

        return preds


class RFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Random-Sampling-Based Fuzzy Min-Max Neural Network Ensemble (RF-FMNN).

    Combines multiple independent FMNN models using bootstrap random sampling,
    optional random feature subspaces (max_features), and deterministic
    majority/plurality voting.

    When n_estimators is None, bootstrap samples are generated until the union
    of unique observation IDs covers the complete training dataset (up to
    max_bags_limit). Hyperboxes are constructed without contraction; overlapping
    regions are resolved via nearest data-based centroids.
    """
    def __init__(
        self,
        theta: float = 0.2,
        gamma: float = 1.0,
        n_estimators: int = None,
        max_features=None,
        max_bags_limit: int = 150,
        prune: bool = False,
        random_state: int = 42,
    ):
        self.theta = theta
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_bags_limit = max_bags_limit
        self.prune = prune
        self.random_state = random_state
        self.models_ = []
        self.feature_subsets_ = []
        self.classes_ = None
        self.n_bags_ = 0

    def get_total_boxes(self) -> int:
        return sum(len(m.hyperboxes) for m in self.models_)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        bag_specs = _generate_bag_specs(
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_bags_limit=self.max_bags_limit,
            random_state=self.random_state,
        )

        self.n_bags_ = len(bag_specs)
        self.models_ = []
        self.feature_subsets_ = []
        all_features = np.arange(n_features)

        for boot_idx, feat_idx in bag_specs:
            X_boot = X[boot_idx] if feat_idx is None else X[np.ix_(boot_idx, feat_idx)]
            y_boot = y[boot_idx]
            model = SingleFMNN(theta=self.theta, gamma=self.gamma, prune=self.prune)
            model.fit(X_boot, y_boot)
            self.models_.append(model)
            self.feature_subsets_.append(all_features if feat_idx is None else feat_idx)

        return self

    def _collect_votes(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=np.int64)
        if n_samples == 0 or not self.models_:
            return votes

        row_idx = np.arange(n_samples)
        n_features = X.shape[1]
        for model, feat_idx in zip(self.models_, self.feature_subsets_):
            X_in = X if len(feat_idx) == n_features else X[:, feat_idx]
            preds = model.predict(X_in)
            col_idx = np.searchsorted(self.classes_, preds)
            votes[row_idx, col_idx] += 1
        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Collect independent predictions and perform deterministic majority voting."""
        X = np.asarray(X, dtype=np.float64)
        if len(X) == 0:
            return np.empty(0, dtype=self.classes_.dtype if self.classes_ is not None else int)
        votes = self._collect_votes(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(X) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        votes = self._collect_votes(X)
        return votes / float(self.n_bags_)
