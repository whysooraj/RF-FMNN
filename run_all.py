#!/usr/bin/env python3
import sys
import os

from experiments.run_benchmark import run_cross_validation_benchmark
from experiments.ablation_study import run_ablation_study
from experiments.visualize import plot_decision_boundaries, plot_benchmark_bars

def main():
    print("================================================================================")
    print("RF-FMNN: COMPREHENSIVE EXPERIMENTAL SUITE & VALIDATION")
    print("================================================================================")

    # 1. Run full 5-fold cross-validation benchmark
    print("\n[PHASE 1/3] Executing 5-Fold Stratified Cross-Validation Benchmark...")
    run_cross_validation_benchmark(n_splits=5, random_state=42)

    # 2. Run ablation study
    print("\n[PHASE 2/3] Executing Architectural Ablation Study...")
    run_ablation_study(random_state=42)

    # 3. Generate visualization plots
    print("\n[PHASE 3/3] Generating Figures and Decision Boundary Visualizations...")
    plot_decision_boundaries()
    plot_benchmark_bars()

    print("\n================================================================================")
    print("ALL EXPERIMENTAL RUNS, BENCHMARKS, AND PLOTS COMPLETED SUCCESSFULLY!")
    print("Figures saved in ./figures/")
    print("Results saved in ./experiments/results/")
    print("================================================================================")

if __name__ == '__main__':
    main()
