import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.rf_fmnn import RFFMNNClassifier
from src.utils.metrics import evaluate_model


def _paired_ttest(scores_a, scores_b):
    """Run paired Student t-test safely, handling zero-variance fold differences."""
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


def run_cross_validation_benchmark(n_splits: int = 5, random_state: int = 42):
    datasets = load_benchmark_datasets(scale=False)
    models = {
        'Basic FMNN (Simpson 1992)': lambda: BaseFMNN(theta=0.2, gamma=1.0),
        'Decision Tree (CART)': lambda: DecisionTreeClassifier(random_state=random_state),
        'Random Forest (Breiman)': lambda: RandomForestClassifier(
            n_estimators=30, random_state=random_state
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

    proposed_variants = [
        'Proposed RF-FMNN (Bagging)',
        'Proposed RF-FMNN (Subspace sqrt)',
    ]
    stat_baselines = [
        'Basic FMNN (Simpson 1992)',
        'Random Forest (Breiman)',
    ]

    results = []
    fold_accuracies = {}

    print("=" * 80)
    print(
        f"STARTING FULL BENCHMARK: 6 Classification Metrics Across "
        f"{len(datasets)} Datasets and {len(models)} Models"
    )
    print("=" * 80)

    for ds_name, ds_info in datasets.items():
        X, y = ds_info['X'], ds_info['y']
        fold_accuracies[ds_name] = {}
        print(
            f"\n---> Dataset: {ds_name} "
            f"(Samples: {len(X)}, Dim: {ds_info['n_features']}, Classes: {ds_info['n_classes']})"
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(X, y))

        for model_name, model_fn in models.items():
            metrics_accum = {
                'accuracy': [],
                'precision': [],
                'recall_sensitivity': [],
                'specificity': [],
                'f1_score': [],
                'auc_roc': [],
                'total_hyperboxes': [],
                'train_time_sec': [],
            }

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = np.clip(scaler.transform(X[test_idx]), 0.0, 1.0)
                y_train, y_test = y[train_idx], y[test_idx]

                m = model_fn()
                metrics = evaluate_model(m, X_train, y_train, X_test, y_test)
                if metrics['total_hyperboxes'] == 0 and hasattr(m, 'get_total_boxes'):
                    metrics['total_hyperboxes'] = int(m.get_total_boxes())

                for k in metrics_accum:
                    metrics_accum[k].append(metrics[k])

            fold_accuracies[ds_name][model_name] = list(metrics_accum['accuracy'])

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
                'mean_train_time': float(np.mean(metrics_accum['train_time_sec'])),
                'vs_basic_fmnn_p_value': None,
                'vs_basic_fmnn_significant': None,
                'vs_random_forest_p_value': None,
                'vs_random_forest_significant': None,
            }

            print(
                f"  {model_name:32s} | Acc: {res_entry['accuracy_mean']*100:5.2f}% | "
                f"Prec: {res_entry['precision_mean']*100:5.2f}% | "
                f"Rec: {res_entry['recall_sensitivity_mean']*100:5.2f}% | "
                f"Spec: {res_entry['specificity_mean']*100:5.2f}% | "
                f"F1: {res_entry['f1_score_mean']*100:5.2f}% | "
                f"AUC: {res_entry['auc_roc_mean']*100:5.2f}%"
            )

            results.append(res_entry)

    # Compute paired Student t-tests across 5 CV fold accuracies
    stat_records = []
    result_lookup = {(r['dataset'], r['model']): r for r in results}

    print("\n" + "=" * 80)
    print("PAIRED STUDENT T-TEST SIGNIFICANCE (5-FOLD CV ACCURACY, alpha = 0.05)")
    print("=" * 80)

    for ds_name in datasets.keys():
        for prop_name in proposed_variants:
            entry = result_lookup[(ds_name, prop_name)]
            for base_name in stat_baselines:
                stats_res = _paired_ttest(
                    fold_accuracies[ds_name][prop_name],
                    fold_accuracies[ds_name][base_name],
                )
                record = {
                    'dataset': ds_name,
                    'proposed_model': prop_name,
                    'baseline_model': base_name,
                    'mean_diff': stats_res['mean_diff'],
                    't_statistic': stats_res['t_statistic'],
                    'p_value': stats_res['p_value'],
                    'significant': stats_res['significant'],
                }
                stat_records.append(record)

                if base_name == 'Basic FMNN (Simpson 1992)':
                    entry['vs_basic_fmnn_p_value'] = stats_res['p_value']
                    entry['vs_basic_fmnn_significant'] = stats_res['significant']
                elif base_name == 'Random Forest (Breiman)':
                    entry['vs_random_forest_p_value'] = stats_res['p_value']
                    entry['vs_random_forest_significant'] = stats_res['significant']

                sig_label = "YES" if stats_res['significant'] else "NO"
                print(
                    f"  {ds_name:13s} | {prop_name:32s} vs {base_name:25s} | "
                    f"diff={stats_res['mean_diff']*100:+6.2f}pp | "
                    f"t={stats_res['t_statistic']:+6.3f} | p={stats_res['p_value']:.4f} | sig={sig_label}"
                )

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open('experiments/results/statistical_significance.json', 'w') as f:
        json.dump(stat_records, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv('experiments/results/benchmark_results.csv', index=False)

    # Generate Markdown Tables by Model
    md_content = "# Comprehensive Benchmark Metrics: Per-Model and Comparative Tables\n\n"
    md_content += (
        "Evaluated using 5-Fold Stratified Cross-Validation with strictly in-fold MinMaxScaler "
        "fitting (test features clipped to [0, 1]) across Accuracy, Precision, Recall (Sensitivity), "
        "Specificity, F1-Score, and AUC-ROC.\n\n"
    )

    for m_name in models.keys():
        md_content += f"## Model: {m_name}\n\n"
        md_content += "| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |\n"
        md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        sub_df = df[df['model'] == m_name]
        for _, row in sub_df.iterrows():
            md_content += (
                f"| **{row['dataset']}** | "
                f"{row['accuracy_mean']*100:.2f}% +/- {row['accuracy_std']*100:.2f}% | "
                f"{row['precision_mean']*100:.2f}% +/- {row['precision_std']*100:.2f}% | "
                f"{row['recall_sensitivity_mean']*100:.2f}% +/- {row['recall_sensitivity_std']*100:.2f}% | "
                f"{row['specificity_mean']*100:.2f}% +/- {row['specificity_std']*100:.2f}% | "
                f"{row['f1_score_mean']*100:.2f}% +/- {row['f1_score_std']*100:.2f}% | "
                f"{row['auc_roc_mean']*100:.2f}% +/- {row['auc_roc_std']*100:.2f}% | "
                f"{row['mean_hyperboxes']:.1f} |\n"
            )
        md_content += "\n"

    # Comparative Summary Table across all 5 models
    md_content += "## Overall Comparative Summary: Accuracy and F1-Score Across Models\n\n"
    md_content += (
        "| Dataset | Metric | Basic FMNN (1992) | CART Tree | Random Forest | "
        "Proposed RF-FMNN (Bagging) | Proposed RF-FMNN (Subspace sqrt) |\n"
    )
    md_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for ds_name in datasets.keys():
        sub_ds = df[df['dataset'] == ds_name]
        acc_vals = {
            r['model']: f"{r['accuracy_mean']*100:.2f}% +/- {r['accuracy_std']*100:.2f}%"
            for _, r in sub_ds.iterrows()
        }
        f1_vals = {
            r['model']: f"{r['f1_score_mean']*100:.2f}% +/- {r['f1_score_std']*100:.2f}%"
            for _, r in sub_ds.iterrows()
        }
        md_content += (
            f"| **{ds_name}** | Accuracy | "
            f"{acc_vals.get('Basic FMNN (Simpson 1992)', '-')} | "
            f"{acc_vals.get('Decision Tree (CART)', '-')} | "
            f"{acc_vals.get('Random Forest (Breiman)', '-')} | "
            f"**{acc_vals.get('Proposed RF-FMNN (Bagging)', '-')}** | "
            f"**{acc_vals.get('Proposed RF-FMNN (Subspace sqrt)', '-')}** |\n"
        )
        md_content += (
            f"| | F1-Score | "
            f"{f1_vals.get('Basic FMNN (Simpson 1992)', '-')} | "
            f"{f1_vals.get('Decision Tree (CART)', '-')} | "
            f"{f1_vals.get('Random Forest (Breiman)', '-')} | "
            f"**{f1_vals.get('Proposed RF-FMNN (Bagging)', '-')}** | "
            f"**{f1_vals.get('Proposed RF-FMNN (Subspace sqrt)', '-')}** |\n"
        )
    md_content += "\n"

    # Statistical Significance Table (Paired Student t-test)
    md_content += "## Statistical Significance (Paired Student t-test on 5-Fold CV Accuracy)\n\n"
    md_content += (
        "Two-sided paired Student t-test (`scipy.stats.ttest_rel`) comparing 5-fold CV accuracies of "
        "each Proposed RF-FMNN variant against Basic FMNN (Simpson 1992) and Random Forest (Breiman). "
        "Identical fold scores (zero variance) yield `p = 1.0000`.\n\n"
    )
    md_content += (
        "| Dataset | Proposed Variant | Baseline Model | Mean Acc Diff (pp) | "
        "t-statistic | p-value | Significant (p < 0.05) |\n"
    )
    md_content += "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for rec in stat_records:
        sig_str = "Yes" if rec['significant'] else "No"
        md_content += (
            f"| **{rec['dataset']}** | {rec['proposed_model']} | {rec['baseline_model']} | "
            f"{rec['mean_diff']*100:+.2f} | {rec['t_statistic']:+.4f} | "
            f"{rec['p_value']:.4f} | {sig_str} |\n"
        )
    md_content += "\n"

    with open('experiments/results/DETAILED_METRICS_TABLES.md', 'w') as f:
        f.write(md_content)

    print("\nDetailed markdown tables generated at experiments/results/DETAILED_METRICS_TABLES.md")
    return df


if __name__ == '__main__':
    run_cross_validation_benchmark()
