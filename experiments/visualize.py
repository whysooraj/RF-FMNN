import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_loader import load_benchmark_datasets
from src.fmnn.base_fmnn import BaseFMNN
from src.fmnn.rf_fmnn import RFFMNNClassifier

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.edgecolor': '#71717a',
    'axes.linewidth': 0.8,
    'grid.color': '#e4e4e7',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'figure.dpi': 300,
})

MODEL_COLORS = {
    'Basic FMNN (Simpson 1992)': '#a1a1aa',
    'Decision Tree (CART)': '#71717a',
    'Random Forest (Breiman)': '#3f3f46',
    'Proposed RF-FMNN (Bagging)': '#0284c7',
    'Proposed RF-FMNN (Subspace sqrt)': '#0369a1',
}


def plot_decision_boundaries():
    datasets = load_benchmark_datasets(scale=True)
    target_datasets = ['TwoMoons', 'Circles']
    models = {
        'Basic FMNN (Simpson 1992)': lambda: BaseFMNN(theta=0.2, gamma=1.0),
        'Proposed RF-FMNN (Bagging)': lambda: RFFMNNClassifier(
            theta=0.2, gamma=1.0, n_estimators=30, max_features=None, random_state=42
        ),
        'Proposed RF-FMNN (Subspace sqrt)': lambda: RFFMNNClassifier(
            theta=0.2, gamma=1.0, n_estimators=30, max_features='sqrt', random_state=42
        ),
    }

    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(len(target_datasets), len(models), figsize=(14, 8))
    fig.patch.set_facecolor('#ffffff')

    for row_idx, ds_name in enumerate(target_datasets):
        X = datasets[ds_name]['X']
        y = datasets[ds_name]['y']

        h = 0.015
        x_min, x_max = X[:, 0].min() - 0.08, X[:, 0].max() + 0.08
        y_min, y_max = X[:, 1].min() - 0.08, X[:, 1].max() + 0.08
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        for col_idx, (model_name, model_fn) in enumerate(models.items()):
            ax = axes[row_idx, col_idx]
            model_inst = model_fn()
            model_inst.fit(X, y)
            Z = model_inst.predict(grid_points).reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
            ax.scatter(
                X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='#18181b', s=22, alpha=0.85
            )

            if row_idx == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold', color='#18181b')
            if col_idx == 0:
                ax.set_ylabel(ds_name, fontsize=11, fontweight='bold', color='#18181b')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_path = 'figures/decision_boundaries.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved decision boundaries plot to {output_path}")


def _plot_metric_bars(df, mean_col, std_col, ylabel, title, output_path):
    datasets = list(df['dataset'].unique())
    models = list(df['model'].unique())

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#ffffff')
    x = np.arange(len(datasets))
    width = 0.78 / max(1, len(models))

    all_vals = []
    for i, model in enumerate(models):
        sub_df = df[df['model'] == model]
        vals = [sub_df[sub_df['dataset'] == ds][mean_col].values[0] * 100.0 for ds in datasets]
        errs = [sub_df[sub_df['dataset'] == ds][std_col].values[0] * 100.0 for ds in datasets]
        all_vals.extend([v - e for v, e in zip(vals, errs)])

        color = MODEL_COLORS.get(model, '#52525b')
        ax.bar(
            x + (i - len(models) / 2.0 + 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=2.5,
            label=model,
            color=color,
            edgecolor='#27272a',
            linewidth=0.5,
            error_kw={'elinewidth': 0.8, 'ecolor': '#27272a'},
        )

    y_low = max(0.0, min(60.0, float(np.floor(min(all_vals) / 10.0) * 10.0 - 5.0)))
    ax.set_ylabel(ylabel, fontsize=11, color='#18181b')
    ax.set_title(title, fontsize=13, fontweight='bold', color='#18181b', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim(y_low, 104.0)
    ax.legend(loc='lower right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")


def plot_benchmark_bars():
    csv_path = 'experiments/results/benchmark_results.csv'
    if not os.path.exists(csv_path):
        print("Run benchmark first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs('figures', exist_ok=True)

    _plot_metric_bars(
        df,
        mean_col='accuracy_mean',
        std_col='accuracy_std',
        ylabel='5-Fold Cross-Validation Accuracy (%)',
        title='Classification Accuracy Across Standard Benchmark Datasets',
        output_path='figures/benchmark_accuracy.png',
    )

    f1_mean_col = 'f1_score_mean' if 'f1_score_mean' in df.columns else 'f1_macro_mean'
    f1_std_col = 'f1_score_std' if 'f1_score_std' in df.columns else 'f1_macro_std'
    _plot_metric_bars(
        df,
        mean_col=f1_mean_col,
        std_col=f1_std_col,
        ylabel='Macro F1-Score (%)',
        title='Macro F1-Score Comparison Across Benchmark Datasets',
        output_path='figures/benchmark_f1.png',
    )


if __name__ == '__main__':
    plot_decision_boundaries()
    plot_benchmark_bars()
