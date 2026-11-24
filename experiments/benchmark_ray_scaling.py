import json
import os
import sys
import time

import numpy as np
import ray
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.ray_rf_fmnn import RayRFFMNNClassifier
from src.fmnn.rf_fmnn import RFFMNNClassifier


def _prepare_leakfree_split(X, y, test_size=0.2, random_state=42):
    """Split raw data first, then fit MinMaxScaler strictly on training split."""
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = np.clip(scaler.transform(X_te_raw), 0.0, 1.0)
    return X_tr, X_te, y_tr, y_te


def _split_into_shards(X, y, num_shards):
    """Split dataset into balanced contiguous shards using array_split."""
    indices = np.array_split(np.arange(len(X)), num_shards)
    return [(X[idx], y[idx]) for idx in indices if len(idx) > 0]


# -----------------------------------------------------------------------------
# Ray Workers for Parallel BaseFMNN Baselines
# -----------------------------------------------------------------------------
@ray.remote
def _fit_base_fmnn_shard(X_chunk, y_chunk, theta, gamma):
    """Worker trains a standard Simpson BaseFMNN on a single data shard."""
    model = BaseFMNN(theta=theta, gamma=gamma)
    model.fit(X_chunk, y_chunk)
    return model


class PartitionedRayBaseFMNN:
    """
    Naive Data-Partitioned Simpson FMNN using Ray (Box Pooling).
    Splits data across p cores, trains local BaseFMNN on each shard,
    and pools hyperboxes together without cross-shard contraction.
    Demonstrates cross-shard collision and hyperbox proliferation.
    """

    def __init__(self, theta=0.2, gamma=1.0, num_cpus=4):
        self.theta = theta
        self.gamma = gamma
        self.num_cpus = num_cpus
        self.hyperboxes = []
        self.classes_ = None

    def fit(self, X, y):
        shards = _split_into_shards(X, y, self.num_cpus)

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        futures = [
            _fit_base_fmnn_shard.remote(X_c, y_c, self.theta, self.gamma)
            for X_c, y_c in shards
        ]
        shard_models = ray.get(futures)

        self.hyperboxes = []
        all_classes = set()
        for model in shard_models:
            self.hyperboxes.extend(model.hyperboxes)
            if model.classes_ is not None:
                all_classes.update(model.classes_)
        self.classes_ = np.array(sorted(all_classes))
        return self

    def get_total_boxes(self):
        return len(self.hyperboxes)

    def predict(self, X):
        if not self.hyperboxes:
            return np.zeros(len(X), dtype=int)

        preds = []
        for x in X:
            containing = [b for b in self.hyperboxes if b.contains(x)]
            candidates = containing if containing else self.hyperboxes
            best_mem = -1.0
            best_c = candidates[0].class_label if candidates else self.classes_[0]
            for b in candidates:
                m = self._membership(x, b)
                if m > best_mem:
                    best_mem = m
                    best_c = b.class_label
            preds.append(best_c)
        return np.array(preds)

    def _membership(self, x, box):
        n = len(x)
        diff_w = x - box.w
        diff_v = box.v - x
        term1 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_w)))
        term2 = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * np.minimum(1.0, diff_v)))
        return float(np.sum(term1 + term2) / (2.0 * n))


class ShardedVotingRayBaseFMNN:
    """
    Fair Non-Strawman Parallel Simpson FMNN Baseline (Sharded Majority Voting).
    Trains p independent BaseFMNN classifiers on p disjoint data shards of size N/p
    and aggregates test predictions via majority voting.
    Avoids uncontracted box-pooling collisions, isolating the accuracy cost of
    reducing per-learner training coverage from N to N/p compared to RF-FMNN's
    full N-size bootstrap bags.
    """

    def __init__(self, theta=0.2, gamma=1.0, num_cpus=4):
        self.theta = theta
        self.gamma = gamma
        self.num_cpus = num_cpus
        self.models_ = []
        self.classes_ = None

    def fit(self, X, y):
        shards = _split_into_shards(X, y, self.num_cpus)

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True, logging_level="ERROR")

        futures = [
            _fit_base_fmnn_shard.remote(X_c, y_c, self.theta, self.gamma)
            for X_c, y_c in shards
        ]
        self.models_ = ray.get(futures)

        all_classes = set()
        for model in self.models_:
            if model.classes_ is not None:
                all_classes.update(model.classes_)
        self.classes_ = np.array(sorted(all_classes))
        return self

    def get_total_boxes(self):
        return sum(len(m.hyperboxes) for m in self.models_)

    def predict(self, X):
        if not self.models_:
            return np.zeros(len(X), dtype=int)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        n_samples = len(X)
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=np.int64)

        for model in self.models_:
            shard_preds = model.predict(X)
            for i, pred in enumerate(shard_preds):
                votes[i, class_to_idx[pred]] += 1

        winner_indices = np.argmax(votes, axis=1)
        return self.classes_[winner_indices]


# -----------------------------------------------------------------------------
# Ray Remote Grid Search Worker for Hyperparameter Tuning
# -----------------------------------------------------------------------------
@ray.remote
def _evaluate_theta_candidate(X_train, y_train, X_val, y_val, theta, gamma, n_estimators):
    """Trains an RF-FMNN with given theta and returns accuracy, train time, and box count."""
    t0 = time.perf_counter()
    clf = RFFMNNClassifier(
        theta=theta,
        gamma=gamma,
        n_estimators=n_estimators,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return theta, acc, t_train, int(clf.get_total_boxes())


# -----------------------------------------------------------------------------
# Main Benchmark Routine
# -----------------------------------------------------------------------------
def run_ray_scaling_experiments():
    print("=" * 80)
    print("STARTING RAY PARALLEL PROCESSING SCALING BENCHMARK (1, 2, 4, 8 CORES)")
    print("=" * 80)

    cores_to_test = [1, 2, 4, 8]
    n_estimators = 32  # Fixed ensemble budget to measure parallel scaling cleanly

    # Prepare 3 leak-free datasets (split raw data FIRST, scale strictly on train)
    # 1. Real: Breast Cancer Wisconsin (N=569, d=30)
    bc = load_breast_cancer()
    X_bc_tr, X_bc_te, y_bc_tr, y_bc_te = _prepare_leakfree_split(bc.data, bc.target)

    # 2. Real High-Dimensional: Digits (N=1797, d=64)
    digits = load_digits()
    X_dig_tr, X_dig_te, y_dig_tr, y_dig_te = _prepare_leakfree_split(digits.data, digits.target)

    # 3. Synthetic Scaled: N=5000, d=20
    X_syn, y_syn = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=14,
        n_classes=2,
        random_state=42,
    )
    X_syn_tr, X_syn_te, y_syn_tr, y_syn_te = _prepare_leakfree_split(X_syn, y_syn)

    primary_scaling_ds = "Synthetic-Scaled (N=5000, d=20)"
    benchmark_datasets = {
        "BreastCancer (N=569, d=30)": (X_bc_tr, X_bc_te, y_bc_tr, y_bc_te),
        "Digits (N=1797, d=64)": (X_dig_tr, X_dig_te, y_dig_tr, y_dig_te),
        primary_scaling_ds: (X_syn_tr, X_syn_te, y_syn_tr, y_syn_te),
    }

    results = {
        "metadata": {
            "host_cpus": os.cpu_count() or 1,
            "n_estimators": n_estimators,
            "primary_dataset": primary_scaling_ds,
            "cores_tested": cores_to_test,
        },
        "core_scaling": {},
        "partitioned_base_fmnn": {},
        "sharded_voting_base_fmnn": {},
        "hyperparameter_sweep": {},
        "estimator_scaling": {},
    }

    # -------------------------------------------------------------------------
    # PART 1: Core Scaling for Proposed RF-FMNN (1, 2, 4, 8 Cores)
    # -------------------------------------------------------------------------
    for dname, (X_tr, X_te, y_tr, y_te) in benchmark_datasets.items():
        print(f"\n--- Testing RF-FMNN Core Scaling on {dname} (M={n_estimators} estimators) ---")
        results["core_scaling"][dname] = {}
        t1_baseline = None

        for c in cores_to_test:
            if ray.is_initialized():
                ray.shutdown()
            ray.init(num_cpus=c, ignore_reinit_error=True, logging_level="ERROR")

            clf = RayRFFMNNClassifier(
                theta=0.2,
                gamma=1.0,
                n_estimators=n_estimators,
                num_cpus=c,
                random_state=42,
            )
            try:
                t0 = time.perf_counter()
                clf.fit(X_tr, y_tr)
                train_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                preds = clf.predict(X_te)
                infer_time = time.perf_counter() - t0

                acc = accuracy_score(y_te, preds)
                total_boxes = int(clf.get_total_boxes())
            finally:
                clf.close()

            if c == 1:
                t1_baseline = train_time
                speedup = 1.0
                efficiency = 100.0
            else:
                speedup = t1_baseline / train_time if train_time > 0 else 0.0
                efficiency = (speedup / c) * 100.0

            print(
                f"Cores: {c:2d} | Train Time: {train_time:6.3f}s | Speedup: {speedup:5.2f}x | "
                f"Efficiency: {efficiency:5.1f}% | Acc: {acc*100:.2f}% | Boxes: {total_boxes}"
            )

            results["core_scaling"][dname][str(c)] = {
                "train_time_s": round(train_time, 4),
                "speedup": round(speedup, 2),
                "efficiency_pct": round(efficiency, 1),
                "infer_time_s": round(infer_time, 4),
                "accuracy": round(acc, 4),
                "total_boxes": total_boxes,
            }

    # -------------------------------------------------------------------------
    # PART 2: Normal BaseFMNN Parallelization Baselines (Naive Pooling & Sharded Voting)
    # -------------------------------------------------------------------------
    print(
        f"\n--- Testing Normal BaseFMNN Parallel Baselines on {primary_scaling_ds} (1, 2, 4, 8 Cores) ---"
    )
    X_tr, X_te, y_tr, y_te = benchmark_datasets[primary_scaling_ds]
    t1_base = None

    for c in cores_to_test:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=c, ignore_reinit_error=True, logging_level="ERROR")

        samples_per_shard = int(len(X_tr) // c)

        if c == 1:
            model = BaseFMNN(theta=0.2, gamma=1.0)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0
            t1_base = train_time
            preds = model.predict(X_te)
            acc = accuracy_score(y_te, preds)
            box_count = len(model.hyperboxes)
            speedup = 1.0

            pool_entry = {
                "train_time_s": round(train_time, 4),
                "speedup": round(speedup, 2),
                "accuracy": round(acc, 4),
                "box_count": box_count,
                "samples_per_shard": samples_per_shard,
            }
            vote_entry = dict(pool_entry)
            print(
                f"BaseFMNN Sequential (1 core) | Train Time: {train_time:6.3f}s | "
                f"Speedup: 1.00x | Test Acc: {acc*100:5.2f}% | Box Count: {box_count}"
            )
        else:
            # 2A. Naive Box-Pooling Partitioned BaseFMNN
            part_model = PartitionedRayBaseFMNN(theta=0.2, gamma=1.0, num_cpus=c)
            t0 = time.perf_counter()
            part_model.fit(X_tr, y_tr)
            pool_time = time.perf_counter() - t0
            pool_preds = part_model.predict(X_te)
            pool_acc = accuracy_score(y_te, pool_preds)
            pool_boxes = int(part_model.get_total_boxes())
            pool_speedup = t1_base / pool_time if pool_time > 0 else 0.0

            pool_entry = {
                "train_time_s": round(pool_time, 4),
                "speedup": round(pool_speedup, 2),
                "accuracy": round(pool_acc, 4),
                "box_count": pool_boxes,
                "samples_per_shard": samples_per_shard,
            }

            # 2B. Fair Sharded Majority-Voting BaseFMNN
            vote_model = ShardedVotingRayBaseFMNN(theta=0.2, gamma=1.0, num_cpus=c)
            t0 = time.perf_counter()
            vote_model.fit(X_tr, y_tr)
            vote_time = time.perf_counter() - t0
            vote_preds = vote_model.predict(X_te)
            vote_acc = accuracy_score(y_te, vote_preds)
            vote_boxes = int(vote_model.get_total_boxes())
            vote_speedup = t1_base / vote_time if vote_time > 0 else 0.0

            vote_entry = {
                "train_time_s": round(vote_time, 4),
                "speedup": round(vote_speedup, 2),
                "accuracy": round(vote_acc, 4),
                "box_count": vote_boxes,
                "samples_per_shard": samples_per_shard,
            }

            print(
                f"BaseFMNN Box-Pool  Cores: {c:2d} | Train: {pool_time:6.3f}s | "
                f"Speedup: {pool_speedup:5.2f}x | Acc: {pool_acc*100:5.2f}% | Boxes: {pool_boxes}"
            )
            print(
                f"BaseFMNN ShardVote Cores: {c:2d} | Train: {vote_time:6.3f}s | "
                f"Speedup: {vote_speedup:5.2f}x | Acc: {vote_acc*100:5.2f}% | Boxes: {vote_boxes} "
                f"(N/p={samples_per_shard})"
            )

        results["partitioned_base_fmnn"][str(c)] = pool_entry
        results["sharded_voting_base_fmnn"][str(c)] = vote_entry

    # -------------------------------------------------------------------------
    # PART 3: Parallel Hyperparameter Optimization (Grid Search over theta)
    # -------------------------------------------------------------------------
    print("\n--- Testing Parallel Hyperparameter Tuning (8 theta candidates) ---")
    theta_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    X_tr_sub, y_tr_sub = X_bc_tr, y_bc_tr
    X_val_sub, y_val_sub = X_bc_te, y_bc_te
    t1_sweep = None

    for c in cores_to_test:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=c, ignore_reinit_error=True, logging_level="ERROR")

        t0 = time.perf_counter()
        futures = [
            _evaluate_theta_candidate.remote(X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, th, 1.0, 16)
            for th in theta_candidates
        ]
        ray.get(futures)
        sweep_time = time.perf_counter() - t0

        if c == 1:
            t1_sweep = sweep_time
            sweep_speedup = 1.0
        else:
            sweep_speedup = t1_sweep / sweep_time if sweep_time > 0 else 0.0

        efficiency_pct = (sweep_speedup / c) * 100.0
        print(
            f"Grid Search Cores: {c:2d} | Sweep Time (8 jobs): {sweep_time:6.3f}s | "
            f"Speedup: {sweep_speedup:5.2f}x | Efficiency: {efficiency_pct:5.1f}%"
        )
        results["hyperparameter_sweep"][str(c)] = {
            "sweep_time_s": round(sweep_time, 4),
            "speedup": round(sweep_speedup, 2),
            "efficiency_pct": round(efficiency_pct, 1),
        }

    # -------------------------------------------------------------------------
    # PART 4: Estimator Scaling Throughput on 8 Cores (M = 8, 16, 32, 64)
    # -------------------------------------------------------------------------
    print("\n--- Testing Estimator Depth Scaling on 8 Cores (M = 8, 16, 32, 64) ---")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=8, ignore_reinit_error=True, logging_level="ERROR")

    estimator_budgets = [8, 16, 32, 64]
    X_tr, X_te, y_tr, y_te = benchmark_datasets[primary_scaling_ds]

    for m in estimator_budgets:
        clf = RayRFFMNNClassifier(
            theta=0.2,
            gamma=1.0,
            n_estimators=m,
            num_cpus=8,
            random_state=42,
        )
        try:
            t0 = time.perf_counter()
            clf.fit(X_tr, y_tr)
            t_fit = time.perf_counter() - t0
            throughput = m / t_fit if t_fit > 0 else 0.0
            preds = clf.predict(X_te)
            acc = accuracy_score(y_te, preds)
            total_boxes = int(clf.get_total_boxes())
        finally:
            clf.close()

        print(
            f"Estimators M: {m:2d} | Train Time: {t_fit:6.3f}s | "
            f"Throughput: {throughput:5.2f} bags/sec | Test Acc: {acc*100:5.2f}% | Boxes: {total_boxes}"
        )
        results["estimator_scaling"][str(m)] = {
            "train_time_s": round(t_fit, 4),
            "throughput_bags_per_sec": round(throughput, 2),
            "accuracy": round(acc, 4),
            "total_boxes": total_boxes,
        }

    ray.shutdown()

    # -------------------------------------------------------------------------
    # PART 5: Save JSON and Generate Markdown Report
    # -------------------------------------------------------------------------
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/ray_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    generate_markdown_report(results)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED! RESULTS WRITTEN TO experiments/results/RAY_PARALLEL_SCALING_REPORT.md")
    print("=" * 80)


def generate_markdown_report(results):
    meta = results.get("metadata", {})
    host_cpus = meta.get("host_cpus", os.cpu_count() or 1)
    n_est = meta.get("n_estimators", 32)
    core_scaling = results["core_scaling"]
    primary_ds = meta.get("primary_dataset", list(core_scaling.keys())[-1])

    rf_primary = core_scaling[primary_ds]
    rf_1 = rf_primary["1"]
    rf_8 = rf_primary["8"]

    pool = results["partitioned_base_fmnn"]
    pool_1 = pool["1"]
    pool_8 = pool["8"]
    pool_acc_delta_pp = (pool_8["accuracy"] - pool_1["accuracy"]) * 100.0
    pool_box_delta = pool_8["box_count"] - pool_1["box_count"]
    pool_box_ratio = pool_8["box_count"] / pool_1["box_count"] if pool_1["box_count"] > 0 else 0.0

    vote = results["sharded_voting_base_fmnn"]
    vote_1 = vote["1"]
    vote_8 = vote["8"]
    vote_acc_delta_pp = (vote_8["accuracy"] - vote_1["accuracy"]) * 100.0
    rf_vs_vote_pp = (rf_8["accuracy"] - vote_8["accuracy"]) * 100.0

    sweep = results["hyperparameter_sweep"]
    sweep_1 = sweep["1"]
    sweep_8 = sweep["8"]

    est_scaling = results["estimator_scaling"]
    est_keys = sorted(est_scaling.keys(), key=int)
    est_min_k, est_max_k = est_keys[0], est_keys[-1]
    est_max = est_scaling[est_max_k]

    all_8core_speedups = [ds_data["8"]["speedup"] for ds_data in core_scaling.values()]
    min_8c_speedup = min(all_8core_speedups)
    max_8c_speedup = max(all_8core_speedups)

    md = []
    md.append("# Ray Parallel Processing Benchmark: 1, 2, 4, and 8 Cores Scaling Analysis\n\n")
    md.append(
        f"Evaluated on a {host_cpus}-core Linux host comparing Standard Sequential FMNN (Simpson 1992), "
        f"Naive Box-Pooling BaseFMNN, Fair Sharded Majority-Voting BaseFMNN, and Ray-Distributed Proposed RF-FMNN "
        f"(with strictly train-only MinMax scaling).\n\n"
    )

    md.append("## 1. Executive Summary\n\n")
    md.append(
        f"- **Proposed RF-FMNN Parallel Scaling**: On `{primary_ds}` ($M={n_est}$ estimators), "
        f"Proposed RF-FMNN reduced wall-clock training time from {rf_1['train_time_s']:.3f}s (1 core) to "
        f"{rf_8['train_time_s']:.3f}s (8 cores), achieving **{rf_8['speedup']:.2f}x speedup** "
        f"({rf_8['efficiency_pct']:.1f}% parallel efficiency) with consistent test accuracy "
        f"({rf_1['accuracy']*100:.2f}% at 1 core vs {rf_8['accuracy']*100:.2f}% at 8 cores). "
        f"Across all {len(core_scaling)} evaluated datasets, 8-core speedup ranged from "
        f"{min_8c_speedup:.2f}x to {max_8c_speedup:.2f}x.\n"
    )
    md.append(
        f"- **Normal BaseFMNN Naive Box Pooling**: Sharding `{primary_ds}` across 8 cores and pooling "
        f"uncontracted hyperboxes achieved {pool_8['speedup']:.2f}x speedup ({pool_1['train_time_s']:.3f}s "
        f"to {pool_8['train_time_s']:.3f}s), shifted test accuracy by {pool_acc_delta_pp:+.2f} pp "
        f"({pool_1['accuracy']*100:.2f}% to {pool_8['accuracy']*100:.2f}%), and expanded total hyperboxes "
        f"from {pool_1['box_count']} to {pool_8['box_count']} ({pool_box_delta:+d} boxes, {pool_box_ratio:.2f}x).\n"
    )
    md.append(
        f"- **Normal BaseFMNN Fair Sharded Majority Voting**: Training 8 independent BaseFMNN models on "
        f"disjoint $N/p$ shards ({vote_8['samples_per_shard']} training samples/shard vs "
        f"{vote_1['samples_per_shard']} full-dataset samples) and combining via majority vote achieved "
        f"{vote_8['speedup']:.2f}x speedup with test accuracy {vote_8['accuracy']*100:.2f}% "
        f"({vote_acc_delta_pp:+.2f} pp vs 1-core BaseFMNN). Compared against Sharded-Voting BaseFMNN at 8 cores, "
        f"Proposed RF-FMNN (using full $N$-size bootstrap bags) achieved {rf_8['accuracy']*100:.2f}% accuracy "
        f"({rf_vs_vote_pp:+.2f} pp difference).\n"
    )
    md.append(
        f"- **Parallel Hyperparameter Sweep**: Evaluating 8 $\\theta$ configurations in parallel reduced "
        f"wall-clock sweep time from {sweep_1['sweep_time_s']:.3f}s (1 core) to {sweep_8['sweep_time_s']:.3f}s "
        f"(8 cores), yielding **{sweep_8['speedup']:.2f}x speedup** ({sweep_8['efficiency_pct']:.1f}% efficiency).\n"
    )
    md.append(
        f"- **Estimator Throughput Scaling (8 Cores)**: Scaling from $M={est_min_k}$ to $M={est_max_k}$ "
        f"estimators on `{primary_ds}` reached **{est_max['throughput_bags_per_sec']:.2f} bags/sec** "
        f"({est_max['train_time_s']:.3f}s wall time, {est_max['accuracy']*100:.2f}% test accuracy).\n\n"
    )

    md.append("## 2. Proposed RF-FMNN Core Scaling: Wall-Clock Time, Speedup, and Efficiency\n\n")
    for dname, data in core_scaling.items():
        md.append(f"### Dataset: {dname} ($M={n_est}$ Estimators)\n\n")
        md.append("| Cores ($p$) | Train Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) | Inference Time (s) | Test Accuracy | Total Boxes |\n")
        md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for c, m in data.items():
            md.append(
                f"| **{c}** | {m['train_time_s']:.3f}s | **{m['speedup']:.2f}x** | "
                f"{m['efficiency_pct']:.1f}% | {m['infer_time_s']:.3f}s | "
                f"{m['accuracy']*100:.2f}% | {m['total_boxes']} |\n"
            )
        md.append("\n")

    md.append(f"## 3. Parallelizing Normal BaseFMNN on {primary_ds}: Naive Box Pooling vs Fair Sharded Voting\n\n")
    md.append("### 3A. Naive Data-Partitioned Box Pooling (Cross-Shard Contraction Failure)\n\n")
    md.append("| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Total Hyperboxes | Box Delta vs 1-Core |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for c, m in pool.items():
        delta = m["box_count"] - pool_1["box_count"]
        delta_str = "Baseline" if c == "1" else f"{delta:+d}"
        md.append(
            f"| **{c}** | {m['samples_per_shard']} | {m['train_time_s']:.3f}s | "
            f"{m['speedup']:.2f}x | **{m['accuracy']*100:.2f}%** | {m['box_count']} | {delta_str} |\n"
        )
    md.append("\n")

    md.append("### 3B. Fair Sharded Majority-Voting BaseFMNN (Reduced Per-Learner Coverage $N/p$)\n\n")
    md.append("| Cores ($p$) | Samples/Shard | Train Time (s) | Speedup ($S_p$) | Test Accuracy | Acc Delta vs 1-Core (pp) | Total Hyperboxes |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for c, m in vote.items():
        acc_diff = (m["accuracy"] - vote_1["accuracy"]) * 100.0
        diff_str = "Baseline" if c == "1" else f"{acc_diff:+.2f}"
        md.append(
            f"| **{c}** | {m['samples_per_shard']} | {m['train_time_s']:.3f}s | "
            f"{m['speedup']:.2f}x | **{m['accuracy']*100:.2f}%** | {diff_str} | {m['box_count']} |\n"
        )
    md.append("\n")

    md.append("## 4. Parallel Hyperparameter Tuning (8 $\\theta$ Configurations)\n\n")
    md.append("| Cores ($p$) | Grid Search Time (s) | Speedup ($S_p$) | Parallel Efficiency ($E_p$) |\n")
    md.append("| :---: | :---: | :---: | :---: |\n")
    for c, m in sweep.items():
        md.append(
            f"| **{c}** | {m['sweep_time_s']:.3f}s | **{m['speedup']:.2f}x** | {m['efficiency_pct']:.1f}% |\n"
        )
    md.append("\n")

    md.append(f"## 5. Estimator Scaling on 8 Cores ({primary_ds} Throughput Analysis)\n\n")
    md.append("| Estimators ($M$) | Train Time (s) | Throughput (Bags/sec) | Test Accuracy | Total Boxes |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: |\n")
    for m_val, m in est_scaling.items():
        md.append(
            f"| **{m_val}** | {m['train_time_s']:.3f}s | **{m['throughput_bags_per_sec']:.2f}** | "
            f"{m['accuracy']*100:.2f}% | {m['total_boxes']} |\n"
        )
    md.append("\n")

    with open("experiments/results/RAY_PARALLEL_SCALING_REPORT.md", "w") as f:
        f.write("".join(md))


if __name__ == "__main__":
    run_ray_scaling_experiments()
