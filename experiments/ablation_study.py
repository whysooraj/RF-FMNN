import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.utils.metrics import evaluate_model


def run_ablation_study(random_state: int = 42):
    """
    Sensitivity and component analysis of Proposed RF-FMNN:
    1. RF-FMNN (theta = 0.1)
    2. RF-FMNN (theta = 0.2)
    3. RF-FMNN (theta = 0.3)
    4. RF-FMNN (theta = 0.4)
    """
    datasets = load_benchmark_datasets(scale=False)
    target_datasets = list(datasets.keys())

    configs = {
        'RF-FMNN (theta = 0.1)': lambda: RFFMNNClassifier(
            theta=0.1, n_estimators=30, random_state=random_state
        ),
        'RF-FMNN (theta = 0.2)': lambda: RFFMNNClassifier(
            theta=0.2, n_estimators=30, random_state=random_state
        ),
        'RF-FMNN (theta = 0.3)': lambda: RFFMNNClassifier(
            theta=0.3, n_estimators=30, random_state=random_state
        ),
        'RF-FMNN (theta = 0.4)': lambda: RFFMNNClassifier(
            theta=0.4, n_estimators=30, random_state=random_state
        ),
    }

    ablation_results = []
    print("\n" + "=" * 80)
    print("RUNNING RF-FMNN PARAMETER SENSITIVITY ABLATION STUDY")
    print("=" * 80)

    for ds_name in target_datasets:
        X, y = datasets[ds_name]['X'], datasets[ds_name]['y']
        print(f"\n---> Dataset: {ds_name}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        folds = list(skf.split(X, y))

        for cfg_name, model_fn in configs.items():
            accs = []
            f1s = []
            for train_idx, test_idx in folds:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = np.clip(scaler.transform(X[test_idx]), 0.0, 1.0)
                y_train, y_test = y[train_idx], y[test_idx]

                m = model_fn()
                metrics = evaluate_model(m, X_train, y_train, X_test, y_test)
                accs.append(metrics['accuracy'])
                f1s.append(metrics.get('f1_score', metrics.get('f1_macro', 0.0)))

            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            print(f"  {cfg_name:30s} | Acc: {mean_acc*100:5.2f}% | F1: {mean_f1*100:5.2f}%")
            ablation_results.append({
                'dataset': ds_name,
                'configuration': cfg_name,
                'accuracy': mean_acc,
                'f1_macro': mean_f1,
            })

    os.makedirs('experiments/results', exist_ok=True)
    df_ab = pd.DataFrame(ablation_results)
    df_ab.to_csv('experiments/results/ablation_results.csv', index=False)
    with open('experiments/results/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print("\nSensitivity study completed and saved.")
    return df_ab


if __name__ == '__main__':
    run_ablation_study()
