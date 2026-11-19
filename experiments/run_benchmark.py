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
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.utils.metrics import evaluate_model

def run_cross_validation_benchmark(n_splits: int = 5, random_state: int = 42):
    datasets = load_benchmark_datasets()
    models = {
        'Basic FMNN (Simpson 1992)': lambda: BaseFMNN(theta=0.2, gamma=1.0),
        'Decision Tree (CART)': lambda: DecisionTreeClassifier(random_state=random_state),
        'Random Forest (Breiman)': lambda: RandomForestClassifier(n_estimators=30, random_state=random_state),
        'Proposed RF-FMNN': lambda: RFFMNNClassifier(theta=0.2, gamma=1.0, random_state=random_state)
    }

    results = []

    print("================================================================================")
    print("STARTING FULL BENCHMARK: 6 Extended Classification Metrics Across 5 Datasets")
    print("================================================================================")

    for ds_name, ds_info in datasets.items():
        X, y = ds_info['X'], ds_info['y']
        print(f"\n---> Dataset: {ds_name} (Samples: {len(X)}, Dim: {ds_info['n_features']}, Classes: {ds_info['n_classes']})")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for model_name, model_fn in models.items():
            metrics_accum = {
                'accuracy': [],
                'precision': [],
                'recall_sensitivity': [],
                'specificity': [],
                'f1_score': [],
                'auc_roc': [],
                'total_hyperboxes': [],
                'train_time_sec': []
            }

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                m = model_fn()
                metrics = evaluate_model(m, X_train, y_train, X_test, y_test)

                for k in metrics_accum.keys():
                    metrics_accum[k].append(metrics[k])

            res_entry = {
                'dataset': ds_name,
                'model': model_name,
                'accuracy_mean': float(np.mean(metrics_accum['accuracy'])),
                'accuracy_std': float(np.std(metrics_accum['accuracy'])),
                'precision_mean': float(np.mean(metrics_accum['precision'])),
                'precision_std': float(np.std(metrics_accum['precision'])),
                'recall_sensitivity_mean': float(np.mean(metrics_accum['recall_sensitivity'])),
                'recall_sensitivity_std': float(np.std(metrics_accum['recall_sensitivity'])),
                'specificity_mean': float(np.mean(metrics_accum['specificity'])),
                'specificity_std': float(np.std(metrics_accum['specificity'])),
                'f1_score_mean': float(np.mean(metrics_accum['f1_score'])),
                'f1_score_std': float(np.std(metrics_accum['f1_score'])),
                'auc_roc_mean': float(np.mean(metrics_accum['auc_roc'])),
                'auc_roc_std': float(np.std(metrics_accum['auc_roc'])),
                'mean_hyperboxes': float(np.mean(metrics_accum['total_hyperboxes'])),
                'mean_train_time': float(np.mean(metrics_accum['train_time_sec']))
            }

            print(f"  {model_name:28s} | Acc: {res_entry['accuracy_mean']*100:5.2f}% | Prec: {res_entry['precision_mean']*100:5.2f}% | Rec: {res_entry['recall_sensitivity_mean']*100:5.2f}% | Spec: {res_entry['specificity_mean']*100:5.2f}% | F1: {res_entry['f1_score_mean']*100:5.2f}% | AUC: {res_entry['auc_roc_mean']*100:5.2f}%")

            results.append(res_entry)

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv('experiments/results/benchmark_results.csv', index=False)

    # Generate Markdown Tables by Model
    md_content = "# Comprehensive Benchmark Metrics: Per-Model and Comparative Tables\n\n"
    md_content += "Evaluated using 5-Fold Stratified Cross-Validation across: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and AUC-ROC.\n\n"

    for m_name in models.keys():
        md_content += f"## Model: {m_name}\n\n"
        md_content += "| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |\n"
        md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        sub_df = df[df['model'] == m_name]
        for _, row in sub_df.iterrows():
            md_content += f"| **{row['dataset']}** | {row['accuracy_mean']*100:.2f}% ± {row['accuracy_std']*100:.2f}% | {row['precision_mean']*100:.2f}% ± {row['precision_std']*100:.2f}% | {row['recall_sensitivity_mean']*100:.2f}% ± {row['recall_sensitivity_std']*100:.2f}% | {row['specificity_mean']*100:.2f}% ± {row['specificity_std']*100:.2f}% | {row['f1_score_mean']*100:.2f}% ± {row['f1_score_std']*100:.2f}% | {row['auc_roc_mean']*100:.2f}% ± {row['auc_roc_std']*100:.2f}% | {row['mean_hyperboxes']:.1f} |\n"
        md_content += "\n"

    # Comparative Summary Table: Proposed RF-FMNN vs Baselines (Accuracy & F1)
    md_content += "## Overall Comparative Summary: Accuracy and F1-Score Across Models\n\n"
    md_content += "| Dataset | Metric | Basic FMNN (1992) | CART Tree | Random Forest | Proposed RF-FMNN |\n"
    md_content += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for ds_name in datasets.keys():
        sub_ds = df[df['dataset'] == ds_name]
        acc_vals = {r['model']: f"{r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}%" for _, r in sub_ds.iterrows()}
        f1_vals = {r['model']: f"{r['f1_score_mean']*100:.2f}% ± {r['f1_score_std']*100:.2f}%" for _, r in sub_ds.iterrows()}
        md_content += f"| **{ds_name}** | Accuracy | {acc_vals.get('Basic FMNN (Simpson 1992)', '-')} | {acc_vals.get('Decision Tree (CART)', '-')} | {acc_vals.get('Random Forest (Breiman)', '-')} | **{acc_vals.get('Proposed RF-FMNN', '-')}** |\n"
        md_content += f"| | F1-Score | {f1_vals.get('Basic FMNN (Simpson 1992)', '-')} | {f1_vals.get('Decision Tree (CART)', '-')} | {f1_vals.get('Random Forest (Breiman)', '-')} | **{f1_vals.get('Proposed RF-FMNN', '-')}** |\n"

    with open('experiments/results/DETAILED_METRICS_TABLES.md', 'w') as f:
        f.write(md_content)

    print("\nDetailed markdown tables generated at experiments/results/DETAILED_METRICS_TABLES.md")
    return df

if __name__ == '__main__':
    run_cross_validation_benchmark()
