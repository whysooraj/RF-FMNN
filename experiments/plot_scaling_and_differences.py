import json
import matplotlib.pyplot as plt
import numpy as np

# Apply clean utilitarian anti-slop style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.edgecolor': '#71717a',
    'axes.linewidth': 0.8,
    'grid.color': '#e4e4e7',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'figure.dpi': 300
})

def plot_ray_scaling():
    with open('experiments/results/ray_scaling_results.json', 'r') as f:
        data = json.load(f)

    cores = [1, 2, 4, 8]
    rf_syn_speedup = [data['core_scaling']['Synthetic-Scaled (N=3000, d=20)'][str(c)]['speedup'] for c in cores]
    rf_syn_time = [data['core_scaling']['Synthetic-Scaled (N=3000, d=20)'][str(c)]['train_time_s'] for c in cores]
    rf_syn_eff = [data['core_scaling']['Synthetic-Scaled (N=3000, d=20)'][str(c)]['efficiency_pct'] for c in cores]
    
    base_speedup = [data['partitioned_base_fmnn'][str(c)]['speedup'] for c in cores]
    base_time = [data['partitioned_base_fmnn'][str(c)]['train_time_s'] for c in cores]
    base_boxes = [data['partitioned_base_fmnn'][str(c)]['box_count'] for c in cores]
    base_acc = [data['partitioned_base_fmnn'][str(c)]['accuracy'] * 100 for c in cores]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: Speedup vs Cores
    ax1 = axs[0, 0]
    ax1.plot(cores, cores, 'k--', label='Ideal Linear Speedup', linewidth=1.2, alpha=0.7)
    ax1.plot(cores, rf_syn_speedup, 'o-', color='#0284c7', label='RF-FMNN (Ray Bagging, N=3000)', linewidth=2.0, markersize=6)
    ax1.plot(cores, base_speedup, 's-', color='#e11d48', label='Normal FMNN (Data Partitioning)', linewidth=1.8, markersize=5)
    ax1.set_xlabel('CPU Cores')
    ax1.set_ylabel('Speedup (Sp = T1 / Tp)')
    ax1.set_title('(A) Multi-Core Scaling Speedup', pad=8)
    ax1.set_xticks(cores)
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')

    # Subplot 2: Wall-Clock Training Time
    ax2 = axs[0, 1]
    ax2.plot(cores, rf_syn_time, 'o-', color='#0284c7', label='RF-FMNN Train Time (N=3000)', linewidth=2.0, markersize=6)
    for c, t in zip(cores, rf_syn_time):
        ax2.annotate(f'{t:.1f}s', (c, t), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax2.set_xlabel('CPU Cores')
    ax2.set_ylabel('Wall-Clock Train Time (seconds)')
    ax2.set_title('(B) RF-FMNN Training Latency Reduction', pad=8)
    ax2.set_xticks(cores)
    ax2.set_yscale('log')
    ax2.grid(True, which='both')
    ax2.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')

    # Subplot 3: Parallel Efficiency (%)
    ax3 = axs[1, 0]
    ax3.plot(cores, [100]*4, 'k--', label='100% Ideal Efficiency', linewidth=1.0, alpha=0.6)
    ax3.plot(cores, rf_syn_eff, 'o-', color='#0284c7', label='RF-FMNN Parallel Efficiency', linewidth=2.0, markersize=6)
    for c, eff in zip(cores, rf_syn_eff):
        ax3.annotate(f'{eff:.1f}%', (c, eff), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
    ax3.set_xlabel('CPU Cores')
    ax3.set_ylabel('Parallel Efficiency Ep (%)')
    ax3.set_title('(C) Parallel Compute Efficiency', pad=8)
    ax3.set_xticks(cores)
    ax3.set_ylim([15, 115])
    ax3.grid(True)
    ax3.legend(frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')

    # Subplot 4: Normal FMNN Failure Mode (Box Explosion & Accuracy Drop)
    ax4 = axs[1, 1]
    color_box = '#d97706'
    color_acc = '#dc2626'
    
    ax4.plot(cores, base_boxes, 's-', color=color_box, label='Normal FMNN Hyperbox Count', linewidth=2.0, markersize=6)
    ax4.set_xlabel('CPU Cores (Data Partitions)')
    ax4.set_ylabel('Total Hyperbox Count', color=color_box)
    ax4.tick_params(axis='y', labelcolor=color_box)
    ax4.set_xticks(cores)
    ax4.grid(True)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(cores, base_acc, '^--', color=color_acc, label='Normal FMNN Test Acc (%)', linewidth=1.8, markersize=6)
    ax4_twin.set_ylabel('Test Accuracy (%)', color=color_acc)
    ax4_twin.tick_params(axis='y', labelcolor=color_acc)
    ax4_twin.set_ylim([90.0, 93.5])

    ax4.set_title('(D) Normal FMNN Failure Under Data Sharding', pad=8)
    
    # Combined legend for ax4
    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')

    plt.tight_layout()
    plt.savefig('figures/ray_multicore_scaling.png', dpi=300)
    plt.close()
    print("Saved figures/ray_multicore_scaling.png")

def plot_conceptual_difference():
    """Visualizes geometric difference: Simpson Contraction vs RF-FMNN Centroid Resolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#ffffff')

    # Left: Simpson 1992 Contraction
    ax1.set_title("Normal FMNN (Simpson 1992)\nLossy Box Contraction", fontsize=11, pad=10)
    # Box 1 (Class 1) contracted
    rect1 = plt.Rectangle((0.15, 0.2), 0.35, 0.5, linewidth=1.5, edgecolor='#2563eb', facecolor='#93c5fd', alpha=0.4, label='Class 1 Box')
    # Box 2 (Class 2) contracted back
    rect2 = plt.Rectangle((0.50, 0.2), 0.35, 0.5, linewidth=1.5, edgecolor='#ea580c', facecolor='#fed7aa', alpha=0.4, label='Class 2 Box')
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    # Contraction boundary line
    ax1.axvline(x=0.50, ymin=0.2, ymax=0.7, color='#dc2626', linestyle='--', linewidth=1.5, label='Forced Split Boundary')
    ax1.plot([0.25, 0.35], [0.35, 0.55], 'bo', markersize=5)
    ax1.plot([0.65, 0.75], [0.35, 0.55], 'ro', markersize=5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.text(0.5, 0.1, "Artificial boundary cut\nOrder sensitive & sequential", ha='center', fontsize=9, color='#71717a')
    ax1.legend(loc='upper right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8')
    ax1.grid(True)

    # Right: Proposed RF-FMNN Uncontracted Overlap + Centroids
    ax2.set_title("Proposed RF-FMNN\nUncontracted Overlap & Centroid Decision", fontsize=11, pad=10)
    # Overlapping Box 1
    rect3 = plt.Rectangle((0.15, 0.2), 0.45, 0.5, linewidth=1.5, edgecolor='#2563eb', facecolor='#93c5fd', alpha=0.35, label='Class 1 Hyperbox')
    # Overlapping Box 2
    rect4 = plt.Rectangle((0.40, 0.2), 0.45, 0.5, linewidth=1.5, edgecolor='#ea580c', facecolor='#fed7aa', alpha=0.35, label='Class 2 Hyperbox')
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)
    # Centroids
    c1 = (0.35, 0.45)
    c2 = (0.65, 0.45)
    ax2.plot(c1[0], c1[1], marker='*', color='#1d4ed8', markersize=12, label='Class 1 Centroid (C1)')
    ax2.plot(c2[0], c2[1], marker='*', color='#c2410c', markersize=12, label='Class 2 Centroid (C2)')
    # Test query inside overlap
    qx = (0.46, 0.48)
    ax2.plot(qx[0], qx[1], 'kd', markersize=8, label='Query Point (x)')
    # Distance arrows
    ax2.annotate('', xy=c1, xytext=qx, arrowprops=dict(arrowstyle='->', color='#1d4ed8', lw=1.2, ls=':'))
    ax2.annotate('', xy=c2, xytext=qx, arrowprops=dict(arrowstyle='->', color='#c2410c', lw=1.2, ls=':'))
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.text(0.5, 0.1, "Natural overlap preserved\nResolved by argmin ||x - C_j||", ha='center', fontsize=9, color='#71717a')
    ax2.legend(loc='upper right', frameon=True, facecolor='#fafafa', edgecolor='#d4d4d8', fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('figures/normal_vs_rf_fmnn_concept.png', dpi=300)
    plt.close()
    print("Saved figures/normal_vs_rf_fmnn_concept.png")

if __name__ == '__main__':
    plot_ray_scaling()
    plot_conceptual_difference()
