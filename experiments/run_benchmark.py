import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.raw_rf_fmnn import RawRFFMNNClassifier
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.utils.metrics import evaluate_model

def run_cross_validation_benchmark(n_splits: int = 5, random_state: int = 42):
    datasets = load_benchmark_datasets()
    models = {
        'Basic FMNN (Simpson 1992)': lambda: BaseFMNN(theta=0.2, gamma=1.0),
        'Decision Tree (CART)': lambda: DecisionTreeClassifier(random_state=random_state),
        'Random Forest (Breiman)': lambda: RandomForestClassifier(n_estimators=35, random_state=random_state),
        'Raw Proposed RF-FMNN': lambda: RawRFFMNNClassifier(theta=0.2, gamma=1.0, random_state=random_state),
        'Upgraded RF-FMNN (Ours)': lambda: RFFMNNClassifier(
            n_estimators=35, theta_range=(0.15, 0.35), max_features='sqrt', random_state=random_state
        )
    }

    results = []

    print("================================================================================")
    print("STARTING BENCHMARK: 5-Fold Stratified Cross-Validation (Upgraded Engine)")
    print("================================================================================")

    for ds_name, ds_info in datasets.items():
        X, y = ds_info['X'], ds_info['y']
        print(f"\n---> Dataset: {ds_name} (Samples: {len(X)}, Dim: {ds_info['n_features']}, Classes: {ds_info['n_classes']})")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for model_name, model_fn in models.items():
            fold_accs = []
            fold_f1s = []
            fold_boxes = []
            fold_train_times = []
            fold_infer_times = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                m = model_fn()
                metrics = evaluate_model(m, X_train, y_train, X_test, y_test)

                fold_accs.append(metrics['accuracy'])
                fold_f1s.append(metrics['f1_macro'])
                fold_boxes.append(metrics['total_hyperboxes'])
                fold_train_times.append(metrics['train_time_sec'])
                fold_infer_times.append(metrics['infer_time_sec'])

            mean_acc = float(np.mean(fold_accs))
            std_acc = float(np.std(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            std_f1 = float(np.std(fold_f1s))
            mean_boxes = float(np.mean(fold_boxes))
            mean_train_time = float(np.mean(fold_train_times))

            print(f"  {model_name:32s} | Acc: {mean_acc*100:5.2f}% (±{std_acc*100:4.2f}%) | F1: {mean_f1*100:5.2f}% | Boxes: {mean_boxes:5.1f}")

            results.append({
                'dataset': ds_name,
                'model': model_name,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'f1_macro_mean': mean_f1,
                'f1_macro_std': std_f1,
                'mean_hyperboxes': mean_boxes,
                'mean_train_time': mean_train_time
            })

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv('experiments/results/benchmark_results.csv', index=False)
    print("\nBenchmark saved to experiments/results/.")
    return df

if __name__ == '__main__':
    run_cross_validation_benchmark()
