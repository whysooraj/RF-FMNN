# Comprehensive Benchmark Metrics: Per-Model and Comparative Tables

Evaluated using 5-Fold Stratified Cross-Validation with strictly in-fold MinMaxScaler fitting (test features clipped to [0, 1]) across Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and AUC-ROC.

## Model: Basic FMNN (Simpson 1992)

| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 93.33% +/- 3.65% | 94.07% +/- 3.47% | 93.33% +/- 3.65% | 96.67% +/- 1.83% | 93.28% +/- 3.70% | 92.90% +/- 2.01% | 18.8 |
| **Wine** | 98.89% +/- 2.22% | 98.88% +/- 2.24% | 99.05% +/- 1.90% | 99.47% +/- 1.07% | 98.90% +/- 2.19% | 99.51% +/- 0.31% | 49.6 |
| **BreastCancer** | 95.79% +/- 2.62% | 96.01% +/- 2.93% | 95.02% +/- 2.79% | 91.98% +/- 3.83% | 95.45% +/- 2.82% | 99.23% +/- 0.83% | 83.4 |
| **TwoMoons** | 99.00% +/- 1.33% | 99.05% +/- 1.26% | 99.00% +/- 1.33% | 98.67% +/- 2.67% | 99.00% +/- 1.33% | 99.93% +/- 0.13% | 21.0 |
| **Circles** | 98.33% +/- 1.05% | 98.41% +/- 0.99% | 98.33% +/- 1.05% | 98.67% +/- 1.63% | 98.33% +/- 1.06% | 99.89% +/- 0.14% | 24.2 |
| **Digits** | 98.11% +/- 0.33% | 98.18% +/- 0.30% | 98.09% +/- 0.33% | 99.79% +/- 0.04% | 98.10% +/- 0.33% | 99.63% +/- 0.10% | 408.8 |

## Model: Decision Tree (CART)

| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 95.33% +/- 3.40% | 95.72% +/- 3.26% | 95.33% +/- 3.40% | 97.67% +/- 1.70% | 95.31% +/- 3.41% | 96.50% +/- 2.55% | 0.0 |
| **Wine** | 89.89% +/- 2.90% | 90.78% +/- 3.35% | 90.02% +/- 2.58% | 94.72% +/- 1.39% | 90.12% +/- 2.80% | 92.37% +/- 1.98% | 0.0 |
| **BreastCancer** | 91.04% +/- 2.79% | 90.80% +/- 2.45% | 90.00% +/- 3.68% | 85.88% +/- 7.37% | 90.28% +/- 3.17% | 90.00% +/- 3.68% | 0.0 |
| **TwoMoons** | 96.67% +/- 2.79% | 96.73% +/- 2.76% | 96.67% +/- 2.79% | 96.00% +/- 3.89% | 96.66% +/- 2.79% | 96.67% +/- 2.79% | 0.0 |
| **Circles** | 96.67% +/- 2.79% | 96.84% +/- 2.64% | 96.67% +/- 2.79% | 96.67% +/- 5.16% | 96.66% +/- 2.80% | 96.67% +/- 2.79% | 0.0 |
| **Digits** | 85.47% +/- 1.38% | 85.76% +/- 1.45% | 85.44% +/- 1.39% | 98.39% +/- 0.15% | 85.48% +/- 1.40% | 91.91% +/- 0.77% | 0.0 |

## Model: Random Forest (Breiman)

| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 95.33% +/- 3.40% | 95.72% +/- 3.26% | 95.33% +/- 3.40% | 97.67% +/- 1.70% | 95.31% +/- 3.41% | 99.53% +/- 0.58% | 0.0 |
| **Wine** | 98.30% +/- 2.28% | 98.36% +/- 2.12% | 98.49% +/- 2.05% | 99.14% +/- 1.19% | 98.40% +/- 2.09% | 99.84% +/- 0.20% | 0.0 |
| **BreastCancer** | 94.91% +/- 2.10% | 94.93% +/- 1.82% | 94.32% +/- 2.82% | 92.00% +/- 6.36% | 94.50% +/- 2.34% | 98.55% +/- 0.78% | 0.0 |
| **TwoMoons** | 97.00% +/- 1.25% | 97.07% +/- 1.25% | 97.00% +/- 1.25% | 97.33% +/- 1.33% | 97.00% +/- 1.25% | 99.53% +/- 0.49% | 0.0 |
| **Circles** | 97.67% +/- 2.00% | 97.76% +/- 1.95% | 97.67% +/- 2.00% | 98.67% +/- 2.67% | 97.66% +/- 2.00% | 99.84% +/- 0.17% | 0.0 |
| **Digits** | 97.44% +/- 0.41% | 97.51% +/- 0.38% | 97.43% +/- 0.42% | 99.72% +/- 0.05% | 97.43% +/- 0.41% | 99.82% +/- 0.09% | 0.0 |

## Model: Proposed RF-FMNN (Bagging)

| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 94.00% +/- 4.90% | 94.34% +/- 4.68% | 94.00% +/- 4.90% | 97.00% +/- 2.45% | 93.96% +/- 4.94% | 99.60% +/- 0.39% | 429.4 |
| **Wine** | 96.63% +/- 3.26% | 96.82% +/- 3.16% | 96.95% +/- 2.77% | 98.34% +/- 1.58% | 96.66% +/- 3.19% | 99.62% +/- 0.75% | 1058.2 |
| **BreastCancer** | 96.84% +/- 1.31% | 96.88% +/- 1.53% | 96.53% +/- 1.51% | 95.29% +/- 3.94% | 96.61% +/- 1.41% | 98.85% +/- 1.07% | 1740.4 |
| **TwoMoons** | 98.00% +/- 1.25% | 98.10% +/- 1.16% | 98.00% +/- 1.25% | 98.00% +/- 2.67% | 98.00% +/- 1.25% | 100.00% +/- 0.00% | 528.4 |
| **Circles** | 98.67% +/- 1.25% | 98.73% +/- 1.17% | 98.67% +/- 1.25% | 99.33% +/- 1.33% | 98.67% +/- 1.25% | 100.00% +/- 0.00% | 619.4 |
| **Digits** | 98.39% +/- 0.48% | 98.43% +/- 0.48% | 98.37% +/- 0.49% | 99.82% +/- 0.05% | 98.37% +/- 0.49% | 99.87% +/- 0.12% | 9441.6 |

## Model: Proposed RF-FMNN (Subspace sqrt)

| Dataset | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC-ROC | Mean Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** | 94.00% +/- 4.90% | 94.34% +/- 4.68% | 94.00% +/- 4.90% | 97.00% +/- 2.45% | 93.96% +/- 4.94% | 99.20% +/- 1.13% | 340.0 |
| **Wine** | 96.03% +/- 5.31% | 96.09% +/- 4.98% | 96.59% +/- 4.57% | 98.04% +/- 2.69% | 96.16% +/- 5.02% | 99.56% +/- 0.77% | 673.6 |
| **BreastCancer** | 96.31% +/- 1.87% | 96.32% +/- 2.35% | 96.01% +/- 1.50% | 94.81% +/- 2.77% | 96.07% +/- 1.95% | 99.42% +/- 0.32% | 1048.0 |
| **TwoMoons** | 90.33% +/- 5.42% | 91.00% +/- 4.67% | 90.33% +/- 5.42% | 90.67% +/- 12.18% | 90.25% +/- 5.55% | 95.57% +/- 2.30% | 277.2 |
| **Circles** | 82.33% +/- 4.03% | 85.60% +/- 2.33% | 82.33% +/- 4.03% | 96.67% +/- 2.11% | 81.84% +/- 4.38% | 96.19% +/- 1.85% | 319.4 |
| **Digits** | 94.44% +/- 1.17% | 94.50% +/- 1.20% | 94.38% +/- 1.18% | 99.38% +/- 0.13% | 94.34% +/- 1.22% | 99.46% +/- 0.22% | 5590.4 |

## Overall Comparative Summary: Accuracy and F1-Score Across Models

| Dataset | Metric | Basic FMNN (1992) | CART Tree | Random Forest | Proposed RF-FMNN (Bagging) | Proposed RF-FMNN (Subspace sqrt) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Iris** | Accuracy | 93.33% +/- 3.65% | 95.33% +/- 3.40% | 95.33% +/- 3.40% | **94.00% +/- 4.90%** | **94.00% +/- 4.90%** |
| | F1-Score | 93.28% +/- 3.70% | 95.31% +/- 3.41% | 95.31% +/- 3.41% | **93.96% +/- 4.94%** | **93.96% +/- 4.94%** |
| **Wine** | Accuracy | 98.89% +/- 2.22% | 89.89% +/- 2.90% | 98.30% +/- 2.28% | **96.63% +/- 3.26%** | **96.03% +/- 5.31%** |
| | F1-Score | 98.90% +/- 2.19% | 90.12% +/- 2.80% | 98.40% +/- 2.09% | **96.66% +/- 3.19%** | **96.16% +/- 5.02%** |
| **BreastCancer** | Accuracy | 95.79% +/- 2.62% | 91.04% +/- 2.79% | 94.91% +/- 2.10% | **96.84% +/- 1.31%** | **96.31% +/- 1.87%** |
| | F1-Score | 95.45% +/- 2.82% | 90.28% +/- 3.17% | 94.50% +/- 2.34% | **96.61% +/- 1.41%** | **96.07% +/- 1.95%** |
| **TwoMoons** | Accuracy | 99.00% +/- 1.33% | 96.67% +/- 2.79% | 97.00% +/- 1.25% | **98.00% +/- 1.25%** | **90.33% +/- 5.42%** |
| | F1-Score | 99.00% +/- 1.33% | 96.66% +/- 2.79% | 97.00% +/- 1.25% | **98.00% +/- 1.25%** | **90.25% +/- 5.55%** |
| **Circles** | Accuracy | 98.33% +/- 1.05% | 96.67% +/- 2.79% | 97.67% +/- 2.00% | **98.67% +/- 1.25%** | **82.33% +/- 4.03%** |
| | F1-Score | 98.33% +/- 1.06% | 96.66% +/- 2.80% | 97.66% +/- 2.00% | **98.67% +/- 1.25%** | **81.84% +/- 4.38%** |
| **Digits** | Accuracy | 98.11% +/- 0.33% | 85.47% +/- 1.38% | 97.44% +/- 0.41% | **98.39% +/- 0.48%** | **94.44% +/- 1.17%** |
| | F1-Score | 98.10% +/- 0.33% | 85.48% +/- 1.40% | 97.43% +/- 0.41% | **98.37% +/- 0.49%** | **94.34% +/- 1.22%** |

## Statistical Significance (Paired Student t-test on 5-Fold CV Accuracy)

Two-sided paired Student t-test (`scipy.stats.ttest_rel`) comparing 5-fold CV accuracies of each Proposed RF-FMNN variant against Basic FMNN (Simpson 1992) and Random Forest (Breiman). Identical fold scores (zero variance) yield `p = 1.0000`.

| Dataset | Proposed Variant | Baseline Model | Mean Acc Diff (pp) | t-statistic | p-value | Significant (p < 0.05) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Iris** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | +0.67 | +0.3430 | 0.7489 | No |
| **Iris** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | -1.33 | -1.0000 | 0.3739 | No |
| **Iris** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | +0.67 | +0.3430 | 0.7489 | No |
| **Iris** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | -1.33 | -1.0000 | 0.3739 | No |
| **Wine** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | -2.25 | -2.1165 | 0.1017 | No |
| **Wine** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | -1.67 | -1.5000 | 0.2080 | No |
| **Wine** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | -2.86 | -0.9558 | 0.3933 | No |
| **Wine** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | -2.27 | -1.3635 | 0.2444 | No |
| **BreastCancer** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | +1.05 | +1.2377 | 0.2835 | No |
| **BreastCancer** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | +1.93 | +2.2732 | 0.0854 | No |
| **BreastCancer** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | +0.52 | +0.4385 | 0.6837 | No |
| **BreastCancer** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | +1.41 | +1.0900 | 0.3370 | No |
| **TwoMoons** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | -1.00 | -1.5000 | 0.2080 | No |
| **TwoMoons** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | +1.00 | +0.8847 | 0.4263 | No |
| **TwoMoons** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | -8.67 | -4.1903 | 0.0138 | Yes |
| **TwoMoons** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | -6.67 | -2.2019 | 0.0925 | No |
| **Circles** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | +0.33 | +0.4082 | 0.7040 | No |
| **Circles** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | +1.00 | +1.1767 | 0.3046 | No |
| **Circles** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | -16.00 | -8.5524 | 0.0010 | Yes |
| **Circles** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | -15.33 | -10.6948 | 0.0004 | Yes |
| **Digits** | Proposed RF-FMNN (Bagging) | Basic FMNN (Simpson 1992) | +0.28 | +1.2893 | 0.2668 | No |
| **Digits** | Proposed RF-FMNN (Bagging) | Random Forest (Breiman) | +0.95 | +4.5451 | 0.0105 | Yes |
| **Digits** | Proposed RF-FMNN (Subspace sqrt) | Basic FMNN (Simpson 1992) | -3.67 | -5.4767 | 0.0054 | Yes |
| **Digits** | Proposed RF-FMNN (Subspace sqrt) | Random Forest (Breiman) | -3.00 | -3.9659 | 0.0166 | Yes |

