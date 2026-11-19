import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import StratifiedKFold
from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.fmnn.base_fmnn import BaseFMNN
from src.utils.metrics import evaluate_model

def run_ablation_study(random_state: int = 42):
    """
    Sensitivity and component analysis of Proposed RF-FMNN:
    1. RF-FMNN (theta = 0.1)
    2. RF-FMNN (theta = 0.2)
    3. RF-FMNN (theta = 0.3)
    4. RF-FMNN (theta = 0.4)
    """
    datasets = load_benchmark_datasets()
    target_datasets = ['Iris', 'Wine', 'BreastCancer', 'TwoMoons', 'Circles']

    configs = {
        'RF-FMNN (theta = 0.1)': lambda: RFFMNNClassifier(theta=0.1, random_state=random_state),
        'RF-FMNN (theta = 0.2)': lambda: RFFMNNClassifier(theta=0.2, random_state=random_state),
        'RF-FMNN (theta = 0.3)': lambda: RFFMNNClassifier(theta=0.3, random_state=random_state),
        'RF-FMNN (theta = 0.4)': lambda: RFFMNNClassifier(theta=0.4, random_state=random_state)
    }

    ablation_results = []
    print("\n================================================================================")
    print("RUNNING RF-FMNN PARAMETER SENSITIVITY ABLATION STUDY")
    print("================================================================================")

    for ds_name in target_datasets:
        X, y = datasets[ds_name]['X'], datasets[ds_name]['y']
        print(f"\n---> Dataset: {ds_name}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for cfg_name, model_fn in configs.items():
            accs = []
            f1s = []
            for train_idx, test_idx in skf.split(X, y):
                m = model_fn()
                metrics = evaluate_model(m, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
                accs.append(metrics['accuracy'])
                f1s.append(metrics['f1_macro'])

            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            print(f"  {cfg_name:30s} | Acc: {mean_acc*100:5.2f}% | F1: {mean_f1*100:5.2f}%")
            ablation_results.append({
                'dataset': ds_name,
                'configuration': cfg_name,
                'accuracy': mean_acc,
                'f1_macro': mean_f1
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
