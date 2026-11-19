import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import StratifiedKFold
from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.fmnn.erf_fmnn import EnhancedRFFMNNClassifier
from src.utils.metrics import evaluate_model

def run_ablation_study(random_state: int = 42):
    """
    Ablation experiments to isolate the contribution of:
    1. Ensemble Bagging Size (Coupon Collector vs. 10 vs. 30 vs. 50 estimators)
    2. Feature Subspace Selection (Full features vs. Sqrt features)
    3. Soft Fuzzy Evidence Voting vs Hard Discrete Majority Voting
    """
    datasets = load_benchmark_datasets()
    target_datasets = ['Iris', 'Wine', 'BreastCancer', 'TwoMoons']

    ablation_configs = {
        '1. Proposed RF-FMNN (Coupon Collector, ~5 bags)': lambda: RFFMNNClassifier(theta=0.2, random_state=random_state),
        '2. RF-FMNN with Fixed M=30 Bags (No Subspace)': lambda: EnhancedRFFMNNClassifier(
            n_estimators=30, theta_range=(0.2, 0.2), max_features=1.0, random_state=random_state
        ),
        '3. RF-FMNN + Multi-Scale Theta [0.1, 0.35]': lambda: EnhancedRFFMNNClassifier(
            n_estimators=30, theta_range=(0.1, 0.35), max_features=1.0, random_state=random_state
        ),
        '4. Full ERF-FMNN (Subspaces + Multi-Scale + Soft-Voting)': lambda: EnhancedRFFMNNClassifier(
            n_estimators=30, theta_range=(0.1, 0.35), max_features='sqrt', random_state=random_state
        )
    }

    ablation_results = []
    print("\n================================================================================")
    print("RUNNING ABLATION STUDY")
    print("================================================================================")

    for ds_name in target_datasets:
        X, y = datasets[ds_name]['X'], datasets[ds_name]['y']
        print(f"\n---> Dataset: {ds_name}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for cfg_name, model_fn in ablation_configs.items():
            accs = []
            f1s = []
            for train_idx, test_idx in skf.split(X, y):
                m = model_fn()
                metrics = evaluate_model(m, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
                accs.append(metrics['accuracy'])
                f1s.append(metrics['f1_macro'])

            mean_acc = np.mean(accs)
            mean_f1 = np.mean(f1s)
            print(f"  {cfg_name:60s} | Acc: {mean_acc*100:5.2f}% | F1: {mean_f1*100:5.2f}%")
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
    print("\nAblation study completed and saved.")
    return df_ab

if __name__ == '__main__':
    run_ablation_study()
