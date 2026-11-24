import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import ray
from src.fmnn.rf_fmnn import SingleFMNN, _generate_bag_specs


@ray.remote
class FMNNWorker:
    """
    Stateful Ray actor that trains its assigned RF-FMNN bootstrap bags from
    zero-copy Plasma shared memory and pins trained SingleFMNN models and
    feature subsets permanently in worker RAM.
    """
    def __init__(self):
        self.models = []
        self.feature_subsets = []

    def train_bags(self, X, y, bag_specs, theta: float, gamma: float) -> int:
        self.models = []
        self.feature_subsets = []
        total_boxes = 0

        for boot_idx, feat_idx in bag_specs:
            X_boot = X[boot_idx] if feat_idx is None else X[np.ix_(boot_idx, feat_idx)]
            y_boot = y[boot_idx]
            model = SingleFMNN(theta=theta, gamma=gamma)
            model.fit(X_boot, y_boot)
            self.models.append(model)
            self.feature_subsets.append(feat_idx)
            total_boxes += len(model.hyperboxes)

        return total_boxes

    def predict_votes(self, X, classes) -> np.ndarray:
        n_samples = len(X)
        n_classes = len(classes)
        votes = np.zeros((n_samples, n_classes), dtype=np.int64)
        if n_samples == 0 or not self.models:
            return votes

        row_idx = np.arange(n_samples)
        for model, feat_idx in zip(self.models, self.feature_subsets):
            X_in = X if feat_idx is None else X[:, feat_idx]
            preds = model.predict(X_in)
            col_idx = np.searchsorted(classes, preds)
            votes[row_idx, col_idx] += 1

        return votes


class RayRFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Stateful-Actor Ray-Distributed RF-FMNN Classifier.

    Distributes bootstrap bags round-robin across persistent FMNNWorker actors.
    Trained SingleFMNN learners remain pinned in worker RAM across fit() and
    predict(), eliminating model serialization over IPC.
    """
    def __init__(
        self,
        theta: float = 0.2,
        gamma: float = 1.0,
        n_estimators: int = None,
        max_features=None,
        max_bags_limit: int = 150,
        num_cpus: int = 4,
        random_state: int = 42,
    ):
        self.theta = theta
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_bags_limit = max_bags_limit
        self.num_cpus = num_cpus
        self.random_state = random_state
        self.workers_ = []
        self.total_boxes_ = 0
        self.n_bags_ = 0
        self.classes_ = None

    def get_total_boxes(self) -> int:
        return int(self.total_boxes_)

    def close(self):
        """Kill stateful Ray worker actors and release worker RAM."""
        if self.workers_:
            if ray.is_initialized():
                for worker in self.workers_:
                    try:
                        ray.kill(worker)
                    except Exception:
                        pass
            self.workers_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.close()

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
        self.total_boxes_ = 0

        if self.n_bags_ == 0:
            return self

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        # Share X and y once via zero-copy Plasma Object Store
        X_ref = ray.put(X)
        y_ref = ray.put(y)

        # Round-robin striping across persistent stateful worker actors
        n_workers = min(self.num_cpus, self.n_bags_)
        worker_bags = [bag_specs[w::n_workers] for w in range(n_workers)]
        self.workers_ = [FMNNWorker.remote() for _ in range(n_workers)]

        # Return only integer box counts; models stay pinned in worker RAM
        box_count_futures = [
            worker.train_bags.remote(X_ref, y_ref, bags, self.theta, self.gamma)
            for worker, bags in zip(self.workers_, worker_bags)
        ]
        self.total_boxes_ = int(sum(ray.get(box_count_futures)))

        return self

    def _collect_votes(self, X: np.ndarray) -> np.ndarray:
        X_ref = ray.put(X)
        vote_futures = [
            worker.predict_votes.remote(X_ref, self.classes_)
            for worker in self.workers_
        ]
        vote_matrices = ray.get(vote_futures)
        return np.sum(vote_matrices, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.workers_ or self.classes_ is None:
            return np.array([])
        if len(X) == 0:
            return np.empty(0, dtype=self.classes_.dtype)

        total_votes = self._collect_votes(X)
        return self.classes_[np.argmax(total_votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.workers_ or self.classes_ is None:
            return np.array([])
        if len(X) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)

        total_votes = self._collect_votes(X)
        return total_votes / float(self.n_bags_)
