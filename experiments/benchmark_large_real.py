import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import load_large_real_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.utils.metrics import evaluate_model


def _paired_ttest(scores_a, scores_b):
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    mean_diff = float(np.mean(diff))

    if np.allclose(diff, 0.0) or float(np.std(diff)) == 0.0:
        return {
            'mean_diff': mean_diff,
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
        }

    t_stat, p_val = ttest_rel(a, b)
    if np.isnan(t_stat) or np.isnan(p_val):
        return {
            'mean_diff': mean_diff,
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
        }

    return {
        'mean_diff': mean_diff,
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
    }


def run_large_real_benchmark(n_splits: int = 5, random_state: int = 42):
    print("=" * 80)
    print("EXECUTING LEAK-FREE 5-FOLD CV ON LARGE AUTHENTIC BENCHMARK DATASETS")
    print("Tier 1: MAGIC Gamma Telescope (N=19,020, d=10, C=2)")
    print("Tier 2: Statlog Shuttle (NASA Telemetry, N=58,000, d=9, C=7)")
    print("=" * 80)

    datasets = load_large_real_datasets(scale=False)
    models = {
        'Basic FMNN (Simpson 1992)': lambda: BaseFMNN(theta=0.2, gamma=1.0),
        'Decision Tree (CART)': lambda: DecisionTreeClassifier(random_state=random_state),
        'Random Forest (Breiman)': lambda: RandomForestClassifier(
            n_estimators=30, random_state=random_state, n_jobs=-1
        ),
        'Proposed RF-FMNN (Bagging)': lambda: RFFMNNClassifier(
            theta=0.2,
            gamma=1.0,
            n_estimators=30,
            max_features=None,
            random_state=random_state,
        ),
        'Proposed RF-FMNN (Subspace sqrt)': lambda: RFFMNNClassifier(
            theta=0.2,
            gamma=1.0,
            n_estimators=30,
            max_features='sqrt',
            random_state=random_state,
        ),
    }

    results = {}
    summary_rows = []

    for d_name, d_meta in datasets.items():
        print(f"\n>>> Running 5-Fold Benchmark on {d_name} ({d_meta['description']})...")
        X, y = d_meta['X'], d_meta['y']

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        d_results = {m_name: {
            'accuracy': [], 'precision': [], 'recall': [], 'specificity': [],
            'f1_score': [], 'auc_roc': [], 'train_time': [], 'inference_time': [],
            'hyperboxes': []
        } for m_name in models.keys()}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Leak-free normalization: fit scaler strictly on training split
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            X_train = scaler.fit_transform(X_train_raw)
            X_test = np.clip(scaler.transform(X_test_raw), 0.0, 1.0)

            print(f"  Fold {fold+1}/{n_splits}:")
            for m_name, m_ctor in models.items():
                clf = m_ctor()
                metrics = evaluate_model(clf, X_train, y_train, X_test, y_test)

                d_results[m_name]['accuracy'].append(metrics['accuracy'])
                d_results[m_name]['precision'].append(metrics['precision'])
                d_results[m_name]['recall'].append(metrics['recall_sensitivity'])
                d_results[m_name]['specificity'].append(metrics['specificity'])
                d_results[m_name]['f1_score'].append(metrics['f1_score'])
                d_results[m_name]['auc_roc'].append(metrics['auc_roc'])
                d_results[m_name]['train_time'].append(metrics['train_time_sec'])
                d_results[m_name]['inference_time'].append(metrics['infer_time_sec'])
                d_results[m_name]['hyperboxes'].append(metrics['total_hyperboxes'])
                print(f"    {m_name:32s} -> Acc: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%, Time: {metrics['train_time_sec']:.2f}s, Boxes: {metrics['total_hyperboxes']}")

        results[d_name] = d_results

        # Aggregate metrics
        for m_name in models.keys():
            res = d_results[m_name]
            summary_rows.append({
                'Dataset': d_name,
                'Model': m_name,
                'Accuracy (%)': f"{np.mean(res['accuracy'])*100:.2f} ± {np.std(res['accuracy'])*100:.2f}",
                'Precision (%)': f"{np.mean(res['precision'])*100:.2f} ± {np.std(res['precision'])*100:.2f}",
                'Recall (%)': f"{np.mean(res['recall'])*100:.2f} ± {np.std(res['recall'])*100:.2f}",
                'Specificity (%)': f"{np.mean(res['specificity'])*100:.2f} ± {np.std(res['specificity'])*100:.2f}",
                'F1-Score (%)': f"{np.mean(res['f1_score'])*100:.2f} ± {np.std(res['f1_score'])*100:.2f}",
                'AUC-ROC (%)': f"{np.mean(res['auc_roc'])*100:.2f} ± {np.std(res['auc_roc'])*100:.2f}",
                'Mean Boxes': f"{np.mean(res['hyperboxes']):.1f}",
                'Train Time (s)': f"{np.mean(res['train_time']):.3f}",
                'Infer Time (s)': f"{np.mean(res['inference_time']):.3f}",
            })

    # Statistical significance t-tests against Basic FMNN and Random Forest
    ttest_results = {}
    for d_name in datasets.keys():
        ttest_results[d_name] = {}
        rf_fmnn_bag = results[d_name]['Proposed RF-FMNN (Bagging)']['f1_score']
        rf_fmnn_sub = results[d_name]['Proposed RF-FMNN (Subspace sqrt)']['f1_score']
        base_fmnn = results[d_name]['Basic FMNN (Simpson 1992)']['f1_score']
        breiman_rf = results[d_name]['Random Forest (Breiman)']['f1_score']

        ttest_results[d_name]['RF_FMNN_Bag_vs_BaseFMNN'] = _paired_ttest(rf_fmnn_bag, base_fmnn)
        ttest_results[d_name]['RF_FMNN_Bag_vs_BreimanRF'] = _paired_ttest(rf_fmnn_bag, breiman_rf)
        ttest_results[d_name]['RF_FMNN_Sub_vs_BaseFMNN'] = _paired_ttest(rf_fmnn_sub, base_fmnn)
        ttest_results[d_name]['RF_FMNN_Sub_vs_BreimanRF'] = _paired_ttest(rf_fmnn_sub, breiman_rf)

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("LARGE REAL DATASET 5-FOLD BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    os.makedirs('experiments/results', exist_ok=True)
    out_json = {
        'summary': summary_rows,
        'ttest_significance': ttest_results,
        'raw_metrics': results,
    }
    with open('experiments/results/large_real_benchmark_results.json', 'w') as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved results to experiments/results/large_real_benchmark_results.json")
    return df_summary


if __name__ == '__main__':
    run_large_real_benchmark()
