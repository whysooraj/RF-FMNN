import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import ray
from src.fmnn.rf_fmnn import SingleFMNN

@ray.remote
def _fit_bag_chunk_task(X, y, bag_chunk, theta, gamma):
    """
    Ray remote task to train a chunk of bootstrap FMNN learners.
    Reduces task dispatch overhead and IPC serialization round-trips.
    """
    chunk_models = []
    for boot_idx in bag_chunk:
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        model = SingleFMNN(theta=theta, gamma=gamma)
        model.fit(X_boot, y_boot)
        chunk_models.append(model)
    return chunk_models

@ray.remote
def _predict_chunk_task(models_chunk, X):
    """Ray remote task to compute vectorized predictions for a chunk of models."""
    return [model.predict(X) for model in models_chunk]

class RayRFFMNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Optimized Ray-Distributed RF-FMNN Classifier.
    
    Uses chunked task dispatch, zero-copy Plasma object store,
    and vectorized batch prediction for maximum multi-core efficiency.
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

        # Chunk bags across available workers to maximize parallel efficiency
        n_workers = min(self.num_cpus, self.n_bags_)
        indices = np.array_split(np.arange(self.n_bags_), n_workers)
        bag_chunks = [[bootstrap_bags[i] for i in chunk_idx] for chunk_idx in indices if len(chunk_idx) > 0]

        # Dispatch chunked parallel tasks
        futures = [
            _fit_bag_chunk_task.remote(X_ref, y_ref, chunk, self.theta, self.gamma)
            for chunk in bag_chunks
        ]

        # Gather trained models from worker chunks
        chunk_results = ray.get(futures)
        self.models_ = []
        for chunk_models in chunk_results:
            self.models_.extend(chunk_models)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.models_) == 0:
            return np.array([])

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        X_ref = ray.put(X)
        n_workers = min(self.num_cpus, len(self.models_))
        model_indices = np.array_split(np.arange(len(self.models_)), n_workers)
        model_chunks = [[self.models_[i] for i in chunk_idx] for chunk_idx in model_indices if len(chunk_idx) > 0]

        pred_futures = [_predict_chunk_task.remote(chunk, X_ref) for chunk in model_chunks]
        chunk_preds = ray.get(pred_futures)

        all_preds = []
        for preds_sub in chunk_preds:
            all_preds.extend(preds_sub)
        all_preds = np.array(all_preds)  # Shape: (M, n_samples)

        # Vectorized plurality vote across learners
        n_samples = len(X)
        final_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            sample_votes = all_preds[:, i]
            counts = Counter(sample_votes)
            final_preds[i] = counts.most_common(1)[0][0]

        return final_preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.models_) == 0:
            return np.array([])

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        X_ref = ray.put(X)
        n_workers = min(self.num_cpus, len(self.models_))
        model_indices = np.array_split(np.arange(len(self.models_)), n_workers)
        model_chunks = [[self.models_[i] for i in chunk_idx] for chunk_idx in model_indices if len(chunk_idx) > 0]

        pred_futures = [_predict_chunk_task.remote(chunk, X_ref) for chunk in model_chunks]
        chunk_preds = ray.get(pred_futures)

        all_preds = []
        for preds_sub in chunk_preds:
            all_preds.extend(preds_sub)
        all_preds = np.array(all_preds)

        n_samples = len(X)
        probs = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}

        for i in range(n_samples):
            sample_votes = all_preds[:, i]
            counts = Counter(sample_votes)
            for c, cnt in counts.items():
                if c in class_to_idx:
                    probs[i, class_to_idx[c]] = cnt / self.n_bags_

        return probs
