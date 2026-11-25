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


def load_large_real_datasets(scale: bool = False, cache_dir: str = 'data/cache'):
    """
    Loads large authentic real-world benchmark datasets:
    - Tier 1: MAGIC Gamma Telescope (N=19,020, d=10, C=2, UCI #159 / OpenML #1120)
    - Tier 2: Statlog Shuttle (N=58,000, d=9, C=7, NASA telemetry, OpenML #40685)

    Returns raw unscaled features by default (scale=False) for leak-free cross-validation.
    """
    import io
    import os
    import urllib.request
    import pandas as pd
    from sklearn.datasets import fetch_openml

    os.makedirs(cache_dir, exist_ok=True)

    def _prep(X):
        X_arr = np.asarray(X, dtype=np.float64)
        return MinMaxScaler(feature_range=(0.0, 1.0)).fit_transform(X_arr) if scale else X_arr

    datasets = {}

    # 1. Tier 1: MAGIC Gamma Telescope (UCI #159)
    magic_cache = os.path.join(cache_dir, 'magic04.npz')
    if os.path.exists(magic_cache):
        loaded = np.load(magic_cache)
        X_magic, y_magic = loaded['X'], loaded['y']
    else:
        url_magic = 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
        req = urllib.request.Request(url_magic, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        df_magic = pd.read_csv(io.BytesIO(data), header=None)
        X_magic = df_magic.iloc[:, :-1].values.astype(np.float64)
        y_magic = (df_magic.iloc[:, -1].values == 'g').astype(np.int64)
        np.savez_compressed(magic_cache, X=X_magic, y=y_magic)

    X_magic = _prep(X_magic)
    datasets['MagicGamma'] = {
        'X': X_magic,
        'y': y_magic,
        'n_features': X_magic.shape[1],
        'n_classes': len(np.unique(y_magic)),
        'description': 'MAGIC Gamma Telescope (10 features, 2 classes, N=19,020)'
    }

    # 2. Tier 2: Statlog Shuttle (NASA Telemetry)
    shuttle_cache = os.path.join(cache_dir, 'shuttle.npz')
    if os.path.exists(shuttle_cache):
        loaded = np.load(shuttle_cache)
        X_shuttle, y_shuttle = loaded['X'], loaded['y']
    else:
        shuttle_data = fetch_openml('shuttle', version=1, as_frame=False, parser='auto')
        X_shuttle = shuttle_data.data.astype(np.float64)
        y_raw = shuttle_data.target.astype(np.int64)
        y_shuttle = y_raw - 1 if y_raw.min() == 1 else y_raw
        np.savez_compressed(shuttle_cache, X=X_shuttle, y=y_shuttle)

    X_shuttle = _prep(X_shuttle)
    datasets['Shuttle'] = {
        'X': X_shuttle,
        'y': y_shuttle,
        'n_features': X_shuttle.shape[1],
        'n_classes': len(np.unique(y_shuttle)),
        'description': 'Statlog Shuttle NASA Telemetry (9 features, 7 classes, N=58,000)'
    }

    return datasets

