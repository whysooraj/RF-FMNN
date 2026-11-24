import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# Clean utilitarian anti-slop style (neutral zinc/slate base + single sky accent)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'axes.edgecolor': '#71717a',
    'axes.linewidth': 0.8,
    'grid.color': '#e4e4e7',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'figure.dpi': 300,
})


def _select_largest_dataset(data):
    """Select the primary/largest dataset key dynamically from core_scaling."""
    core_scaling = data.get('core_scaling', {})
    if not core_scaling:
        raise ValueError("ray_scaling_results.json has no 'core_scaling' entries.")

    meta_primary = data.get('metadata', {}).get('primary_dataset')
    if meta_primary and meta_primary in core_scaling:
        return meta_primary

    def _extract_n(name):
        match = re.search(r'N\s*=\s*(\d+)', name)
        return int(match.group(1)) if match else -1

    return max(core_scaling.keys(), key=_extract_n)


def plot_ray_scaling():
    with open('experiments/results/ray_scaling_results.json', 'r') as f:
        data = json.load(f)

    os.makedirs('figures', exist_ok=True)

    core_scaling = data['core_scaling']
    primary_ds = _select_largest_dataset(data)
    cores = sorted(int(c) for c in core_scaling[primary_ds].keys())

    rf_primary_speedup = [core_scaling[primary_ds][str(c)]['speedup'] for c in cores]
    rf_primary_eff = [core_scaling[primary_ds][str(c)]['efficiency_pct'] for c in cores]

    pool_data = data.get('partitioned_base_fmnn', {})
    vote_data = data.get('sharded_voting_base_fmnn', {})
    has_vote = bool(vote_data)

    base_speedup = [pool_data[str(c)]['speedup'] for c in cores]
    base_boxes = [pool_data[str(c)]['box_count'] for c in cores]
    base_acc = [pool_data[str(c)]['accuracy'] * 100.0 for c in cores]

    if has_vote:
        vote_speedup = [vote_data[str(c)]['speedup'] for c in cores]
        vote_acc = [vote_data[str(c)]['accuracy'] * 100.0 for c in cores]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.patch.set_facecolor('#ffffff')

    accent = '#0284c7'
    slate_dark = '#27272a'
    slate_mid = '#52525b'
    slate_light = '#a1a1aa'
    alert_red = '#dc2626'
    alert_amber = '#d97706'

    # Subplot (A): Multi-Core Scaling Speedup
    ax1 = axs[0, 0]
    ax1.plot(cores, cores, '--', color=slate_dark, label='Ideal Linear Speedup', linewidth=1.2, alpha=0.65)
    ax1.plot(
        cores,
        rf_primary_speedup,
        'o-',
        color=accent,
        label=f'Proposed RF-FMNN ({primary_ds})',
        linewidth=2.2,
        markersize=6,
    )
    ax1.plot(
        cores,
        base_speedup,
        's--',
        color=alert_red,
        label='Normal FMNN (Naive Box Pooling)',
        linewidth=1.6,
        markersize=5,
    )
    if has_vote:
        ax1.plot(
            cores,
            vote_speedup,
            '^:',
            color=slate_mid,
            label='Normal FMNN (Sharded Voting)',
            linewidth=1.7,
            markersize=5.5,
        )
    ax1.set_xlabel('CPU Cores')
    ax1.set_ylabel('Speedup (Sp = T1 / Tp)')
    ax1.set_title('(A) Multi-Core Scaling Speedup', pad=8, color=slate_dark, fontweight='bold')
    ax1.set_xticks(cores)
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8', loc='upper left')

    # Subplot (B): Wall-Clock Training Time across all evaluated datasets
    ax2 = axs[0, 1]
    secondary_styles = [
        (slate_light, '^--', 1.5, 5),
        (slate_mid, 's-', 1.6, 5),
        ('#71717a', 'd:', 1.5, 5),
    ]
    sec_idx = 0
    for dname, dmetrics in core_scaling.items():
        d_times = [dmetrics[str(c)]['train_time_s'] for c in cores]
        if dname == primary_ds:
            ax2.plot(cores, d_times, 'o-', color=accent, label=dname, linewidth=2.2, markersize=6)
            for c, t in zip(cores, d_times):
                ax2.annotate(
                    f'{t:.2f}s',
                    (c, t),
                    textcoords='offset points',
                    xytext=(0, 7),
                    ha='center',
                    fontsize=8,
                    color=slate_dark,
                )
        else:
            col, fmt, lw, ms = secondary_styles[sec_idx % len(secondary_styles)]
            sec_idx += 1
            ax2.plot(cores, d_times, fmt, color=col, label=dname, linewidth=lw, markersize=ms)

    ax2.set_xlabel('CPU Cores')
    ax2.set_ylabel('Wall-Clock Train Time (seconds)')
    ax2.set_title('(B) Proposed RF-FMNN Training Latency', pad=8, color=slate_dark, fontweight='bold')
    ax2.set_xticks(cores)
    ax2.set_yscale('log')
    ax2.grid(True, which='both')
    ax2.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8', loc='upper right')

    # Subplot (C): Parallel Compute Efficiency (%)
    ax3 = axs[1, 0]
    ax3.plot(cores, [100.0] * len(cores), '--', color=slate_dark, label='100% Ideal Efficiency', linewidth=1.0, alpha=0.6)
    ax3.plot(
        cores,
        rf_primary_eff,
        'o-',
        color=accent,
        label=f'RF-FMNN Efficiency ({primary_ds})',
        linewidth=2.2,
        markersize=6,
    )
    for c, eff in zip(cores, rf_primary_eff):
        ax3.annotate(
            f'{eff:.1f}%',
            (c, eff),
            textcoords='offset points',
            xytext=(0, 6),
            ha='center',
            fontsize=8,
            color=slate_dark,
        )

    all_effs = list(rf_primary_eff)
    sec_idx = 0
    for dname, dmetrics in core_scaling.items():
        if dname == primary_ds:
            continue
        d_eff = [dmetrics[str(c)]['efficiency_pct'] for c in cores]
        all_effs.extend(d_eff)
        col, fmt, lw, ms = secondary_styles[sec_idx % len(secondary_styles)]
        sec_idx += 1
        ax3.plot(cores, d_eff, fmt, color=col, label=dname, linewidth=lw, markersize=ms, alpha=0.85)

    eff_low = max(0.0, min(all_effs) - 12.0)
    eff_high = max(112.0, max(all_effs) + 12.0)
    ax3.set_xlabel('CPU Cores')
    ax3.set_ylabel('Parallel Efficiency Ep (%)')
    ax3.set_title('(C) Parallel Compute Efficiency', pad=8, color=slate_dark, fontweight='bold')
    ax3.set_xticks(cores)
    ax3.set_ylim([eff_low, eff_high])
    ax3.grid(True)
    ax3.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8', loc='lower left')

    # Subplot (D): Normal FMNN Under Data Sharding (Naive Pooling vs Sharded Voting)
    ax4 = axs[1, 1]
    ax4.plot(
        cores,
        base_boxes,
        's-',
        color=alert_amber,
        label='Box Pooling Hyperbox Count',
        linewidth=2.0,
        markersize=6,
    )
    ax4.set_xlabel('CPU Cores (Data Partitions)')
    ax4.set_ylabel('Total Hyperbox Count', color=alert_amber)
    ax4.tick_params(axis='y', labelcolor=alert_amber)
    ax4.set_xticks(cores)
    ax4.grid(True)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        cores,
        base_acc,
        '^--',
        color=alert_red,
        label='Box Pooling Acc (%)',
        linewidth=1.8,
        markersize=6,
    )
    acc_series = list(base_acc)
    if has_vote:
        ax4_twin.plot(
            cores,
            vote_acc,
            'o-',
            color=slate_dark,
            label='Sharded Voting Acc (%)',
            linewidth=1.8,
            markersize=5.5,
        )
        acc_series.extend(vote_acc)

    acc_min, acc_max = min(acc_series), max(acc_series)
    acc_pad = max(1.0, (acc_max - acc_min) * 0.25)
    ax4_twin.set_ylabel('Test Accuracy (%)', color=slate_dark)
    ax4_twin.tick_params(axis='y', labelcolor=slate_dark)
    ax4_twin.set_ylim([max(0.0, acc_min - acc_pad), min(100.5, acc_max + acc_pad)])

    ax4.set_title('(D) Normal FMNN Under Data Sharding', pad=8, color=slate_dark, fontweight='bold')

    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc='best',
        frameon=True,
        facecolor='#fafafa',
        edgecolor='#d4d4d8',
    )

    plt.tight_layout()
    plt.savefig('figures/ray_multicore_scaling.png', dpi=300)
    plt.close()
    print("Saved figures/ray_multicore_scaling.png")


def plot_conceptual_difference():
    """Visualizes geometric difference: Simpson Contraction vs RF-FMNN Centroid Resolution."""
    os.makedirs('figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.patch.set_facecolor('#ffffff')

    slate_dark = '#27272a'
    slate_muted = '#52525b'
    class1_edge = '#0284c7'
    class1_fill = '#bae6fd'
    class2_edge = '#52525b'
    class2_fill = '#d4d4d8'
    split_red = '#dc2626'

    # Left: Simpson 1992 Contraction
    ax1.set_title("Normal FMNN (Simpson 1992)\nLossy Box Contraction", fontsize=11, pad=10, color=slate_dark, fontweight='bold')
    rect1 = plt.Rectangle((0.15, 0.2), 0.35, 0.5, linewidth=1.5, edgecolor=class1_edge, facecolor=class1_fill, alpha=0.45, label='Class 1 Box')
    rect2 = plt.Rectangle((0.50, 0.2), 0.35, 0.5, linewidth=1.5, edgecolor=class2_edge, facecolor=class2_fill, alpha=0.55, label='Class 2 Box')
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    ax1.axvline(x=0.50, ymin=0.2, ymax=0.7, color=split_red, linestyle='--', linewidth=1.5, label='Forced Split Boundary')
    ax1.plot([0.25, 0.35], [0.35, 0.55], 'o', color=class1_edge, markersize=5)
    ax1.plot([0.65, 0.75], [0.35, 0.55], 's', color=slate_dark, markersize=5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.text(0.5, 0.08, "Artificial boundary cut\nOrder sensitive and sequential", ha='center', fontsize=9, color=slate_muted)
    ax1.legend(loc='upper right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')
    ax1.grid(True)

    # Right: Proposed RF-FMNN Uncontracted Overlap + Centroids
    ax2.set_title("Proposed RF-FMNN\nUncontracted Overlap and Centroid Decision", fontsize=11, pad=10, color=slate_dark, fontweight='bold')
    rect3 = plt.Rectangle((0.15, 0.2), 0.45, 0.5, linewidth=1.5, edgecolor=class1_edge, facecolor=class1_fill, alpha=0.4, label='Class 1 Hyperbox')
    rect4 = plt.Rectangle((0.40, 0.2), 0.45, 0.5, linewidth=1.5, edgecolor=class2_edge, facecolor=class2_fill, alpha=0.45, label='Class 2 Hyperbox')
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)
    c1 = (0.35, 0.45)
    c2 = (0.65, 0.45)
    ax2.plot(c1[0], c1[1], marker='*', color='#0369a1', markersize=12, label='Class 1 Centroid (C1)')
    ax2.plot(c2[0], c2[1], marker='*', color=slate_dark, markersize=12, label='Class 2 Centroid (C2)')
    qx = (0.46, 0.48)
    ax2.plot(qx[0], qx[1], 'd', color=split_red, markersize=7, label='Query Point (x)')
    ax2.annotate('', xy=c1, xytext=qx, arrowprops=dict(arrowstyle='->', color='#0369a1', lw=1.2, ls=':'))
    ax2.annotate('', xy=c2, xytext=qx, arrowprops=dict(arrowstyle='->', color=slate_dark, lw=1.2, ls=':'))
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.text(0.5, 0.08, "Natural overlap preserved\nResolved by argmin ||x - C_j||", ha='center', fontsize=9, color=slate_muted)
    ax2.legend(loc='upper right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8', fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('figures/normal_vs_rf_fmnn_concept.png', dpi=300)
    plt.close()
    print("Saved figures/normal_vs_rf_fmnn_concept.png")


if __name__ == '__main__':
    plot_ray_scaling()
    plot_conceptual_difference()
