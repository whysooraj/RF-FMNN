#!/usr/bin/env python3
from experiments.ablation_study import run_ablation_study
from experiments.benchmark_ray_scaling import run_ray_scaling_experiments
from experiments.plot_scaling_and_differences import (
    plot_conceptual_difference,
    plot_ray_scaling,
)
from experiments.run_benchmark import run_cross_validation_benchmark
from experiments.visualize import plot_benchmark_bars, plot_decision_boundaries


def main():
    print("================================================================================")
    print("RF-FMNN: COMPREHENSIVE EXPERIMENTAL SUITE & VALIDATION")
    print("================================================================================")

    print("\n[PHASE 1/5] Executing 5-Fold Stratified Cross-Validation Benchmark...")
    run_cross_validation_benchmark(n_splits=5, random_state=42)

    print("\n[PHASE 2/5] Executing Architectural Ablation Study...")
    run_ablation_study(random_state=42)

    print("\n[PHASE 3/5] Generating Decision Boundary and Benchmark Bar Visualizations...")
    plot_decision_boundaries()
    plot_benchmark_bars()

    print("\n[PHASE 4/5] Executing Ray Multi-Core Parallel Scaling & Pruning Experiments...")
    run_ray_scaling_experiments()

    print("\n[PHASE 5/5] Generating Ray Scaling and Conceptual Comparison Figures...")
    plot_ray_scaling()
    plot_conceptual_difference()

    print("\n================================================================================")
    print("ALL EXPERIMENTAL RUNS, BENCHMARKS, AND PLOTS COMPLETED SUCCESSFULLY!")
    print("Figures saved in ./figures/")
    print("Results saved in ./experiments/results/")
    print("================================================================================")


if __name__ == '__main__':
    main()
