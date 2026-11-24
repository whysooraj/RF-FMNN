import numpy as np
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    make_moons,
    make_circles,
)
from sklearn.preprocessing import MinMaxScaler


def load_benchmark_datasets(scale: bool = False):
    """
    Loads real-world and synthetic benchmark datasets.
    Returns raw unscaled features by default (scale=False) to prevent
    train/test scaling leakage. Set scale=True to normalize features to [0, 1]^d.
    """
    def _prep(X):
        X_arr = np.asarray(X, dtype=np.float64)
        return MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_arr) if scale else X_arr

    datasets = {}

    # 1. Iris
    iris = load_iris()
    X_iris = _prep(iris.data)
    datasets['Iris'] = {
        'X': X_iris,
        'y': iris.target,
        'n_features': X_iris.shape[1],
        'n_classes': len(np.unique(iris.target)),
        'description': 'Fisher Iris dataset (4 features, 3 classes)'
    }

    # 2. Wine
    wine = load_wine()
    X_wine = _prep(wine.data)
    datasets['Wine'] = {
        'X': X_wine,
        'y': wine.target,
        'n_features': X_wine.shape[1],
        'n_classes': len(np.unique(wine.target)),
        'description': 'Wine chemical analysis (13 features, 3 classes)'
    }

    # 3. Breast Cancer Wisconsin
    bc = load_breast_cancer()
    X_bc = _prep(bc.data)
    datasets['BreastCancer'] = {
        'X': X_bc,
        'y': bc.target,
        'n_features': X_bc.shape[1],
        'n_classes': len(np.unique(bc.target)),
        'description': 'Breast Cancer Wisconsin (30 features, 2 classes)'
    }

    # 4. Non-linear Moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_moons = _prep(X_moons)
    datasets['TwoMoons'] = {
        'X': X_moons,
        'y': y_moons,
        'n_features': 2,
        'n_classes': 2,
        'description': 'Interlocking two moons non-linear manifold (2 features, 2 classes)'
    }

    # 5. Concentric Circles
    X_circ, y_circ = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    X_circ = _prep(X_circ)
    datasets['Circles'] = {
        'X': X_circ,
        'y': y_circ,
        'n_features': 2,
        'n_classes': 2,
        'description': 'Concentric circles non-linear manifold (2 features, 2 classes)'
    }

    # 6. Digits
    digits = load_digits()
    X_digits = _prep(digits.data)
    datasets['Digits'] = {
        'X': X_digits,
        'y': digits.target,
        'n_features': X_digits.shape[1],
        'n_classes': len(np.unique(digits.target)),
        'description': 'Optical handwritten digits (64 features, 10 classes)'
    }

    return datasets
