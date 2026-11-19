import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import ray
from src.fmnn.rf_fmnn import SingleFMNN

@ray.remote
def _fit_single_bag_task(X, y, boot_idx, theta, gamma):
    """
    Ray remote task to train a single FMNN learner on a bootstrap sample.
    Zero-copy read from Ray plasma store for X and y.
    """
    X_boot = X[boot_idx]
    y_boot = y[boot_idx]
    model = SingleFMNN(theta=theta, gamma=gamma)
    model.fit(X_boot, y_boot)
    return model

@ray.remote
def _predict_single_model_task(model, X):
    """Ray remote task to compute predictions for one learner."""
    return model.predict(X)

class RayRFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Ray-Distributed RF-FMNN Classifier.
    
    Parallelizes bootstrap bag training and prediction across 1, 2, 4, 8+ CPU cores
    using Ray plasma shared memory and distributed remote tasks.
    """
    def __init__(self, theta: float = 0.2, gamma: float = 1.0, 
                 n_estimators: int = None, max_bags_limit: int = 150, 
                 num_cpus: int = 4, random_state: int = 42):
        self.theta = theta
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_bags_limit = max_bags_limit
        self.num_cpus = num_cpus
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

        # Generate bootstrap indices
        bootstrap_bags = []
        if self.n_estimators is not None:
            # Fixed number of estimators for benchmarking scaling
            for _ in range(self.n_estimators):
                bootstrap_bags.append(rng.choice(n_samples, size=n_samples, replace=True))
        else:
            # Coupon-collector rule until full coverage
            covered_ids = set()
            while len(covered_ids) < n_samples and len(bootstrap_bags) < self.max_bags_limit:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                covered_ids.update(boot_idx)
                bootstrap_bags.append(boot_idx)

        self.n_bags_ = len(bootstrap_bags)

        # Ensure Ray is initialized with specified num_cpus
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        # Put full X and y into Ray Plasma Object Store (shared zero-copy memory across workers)
        X_ref = ray.put(X)
        y_ref = ray.put(y)

        # Dispatch parallel tasks
        futures = [
            _fit_single_bag_task.remote(X_ref, y_ref, boot_idx, self.theta, self.gamma)
            for boot_idx in bootstrap_bags
        ]

        # Gather trained models
        self.models_ = ray.get(futures)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.models_) == 0:
            return np.array([])

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        X_ref = ray.put(X)
        pred_futures = [_predict_single_model_task.remote(model, X_ref) for model in self.models_]
        all_preds = np.array(ray.get(pred_futures))

        final_preds = []
        for i in range(len(X)):
            sample_preds = all_preds[:, i]
            counts = Counter(sample_preds)
            final_preds.append(counts.most_common(1)[0][0])
        return np.array(final_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.models_) == 0:
            return np.array([])

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        X_ref = ray.put(X)
        pred_futures = [_predict_single_model_task.remote(model, X_ref) for model in self.models_]
        all_preds = np.array(ray.get(pred_futures))

        probs = []
        for i in range(len(X)):
            sample_preds = all_preds[:, i]
            counts = Counter(sample_preds)
            prob_vec = [counts.get(c, 0) / self.n_bags_ for c in self.classes_]
            probs.append(prob_vec)
        return np.array(probs)
