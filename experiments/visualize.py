import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.raw_rf_fmnn import RawRFFMNNClassifier
from src.fmnn.rf_fmnn import RFFMNNClassifier

def plot_decision_boundaries():
    datasets = load_benchmark_datasets()
    target_datasets = ['TwoMoons', 'Circles']
    models = {
        'Basic FMNN (Simpson)': BaseFMNN(theta=0.2, gamma=1.0),
        'Raw Proposed RF-FMNN': RawRFFMNNClassifier(theta=0.2, gamma=1.0, random_state=42),
        'Upgraded RF-FMNN (Ours)': RFFMNNClassifier(
            n_estimators=35, theta_range=(0.15, 0.35), max_features=1.0, random_state=42
        )
    }

    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(len(target_datasets), len(models), figsize=(15, 9))

    for row_idx, ds_name in enumerate(target_datasets):
        X = datasets[ds_name]['X']
        y = datasets[ds_name]['y']

        h = 0.015
        x_min, x_max = X[:, 0].min() - 0.08, X[:, 0].max() + 0.08
        y_min, y_max = X[:, 1].min() - 0.08, X[:, 1].max() + 0.08
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        for col_idx, (model_name, model_inst) in enumerate(models.items()):
            ax = axes[row_idx, col_idx]
            model_inst.fit(X, y)
            Z = model_inst.predict(grid_points)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=25, alpha=0.85)

            if row_idx == 0:
                ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}", fontsize=12, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_path = 'figures/decision_boundaries.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved decision boundaries plot to {output_path}")


def plot_benchmark_bars():
    csv_path = 'experiments/results/benchmark_results.csv'
    if not os.path.exists(csv_path):
        print("Run benchmark first.")
        return

    df = pd.read_csv(csv_path)
    datasets = df['dataset'].unique()
    models = df['model'].unique()

    os.makedirs('figures', exist_ok=True)

    # 1. Accuracy Plot
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(datasets))
    width = 0.16

    for i, model in enumerate(models):
        sub_df = df[df['model'] == model]
        accs = [sub_df[sub_df['dataset'] == ds]['accuracy_mean'].values[0] * 100 for ds in datasets]
        errs = [sub_df[sub_df['dataset'] == ds]['accuracy_std'].values[0] * 100 for ds in datasets]
        ax.bar(x + (i - len(models)/2 + 0.5) * width, accs, width, yerr=errs, capsize=3, label=model)

    ax.set_ylabel('5-Fold Cross-Validation Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Across Standard Benchmark Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('figures/benchmark_accuracy.png', dpi=300)
    plt.close()
    print("Saved benchmark accuracy plot to figures/benchmark_accuracy.png")

    # 2. Macro F1 Plot
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(models):
        sub_df = df[df['model'] == model]
        f1s = [sub_df[sub_df['dataset'] == ds]['f1_macro_mean'].values[0] * 100 for ds in datasets]
        errs = [sub_df[sub_df['dataset'] == ds]['f1_macro_std'].values[0] * 100 for ds in datasets]
        ax.bar(x + (i - len(models)/2 + 0.5) * width, f1s, width, yerr=errs, capsize=3, label=model)

    ax.set_ylabel('Macro F1-Score (%)', fontsize=12)
    ax.set_title('Macro F1-Score Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(50, 105)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('figures/benchmark_f1.png', dpi=300)
    plt.close()
    print("Saved benchmark F1 plot to figures/benchmark_f1.png")

if __name__ == '__main__':
    plot_decision_boundaries()
    plot_benchmark_bars()
