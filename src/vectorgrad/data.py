"""
vectorgrad.data
================

Dependency-free synthetic data generation + a minimal batch iterator, so the
whole project runs with nothing but NumPy (no internet access, no dataset
download required to reproduce results).

`make_blobs` reimplements the handful of features of
`sklearn.datasets.make_blobs` that this project needs, so swapping in a real
dataset (e.g. `sklearn.datasets.fetch_openml("mnist_784")` or
`torchvision.datasets.MNIST`) later is a one-line change in
`examples/train_classifier.py` -- the model/training code does not care
where the arrays came from.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def make_blobs(
    n_samples: int = 1500,
    n_features: int = 2,
    centers: int = 4,
    cluster_std: float = 1.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian blobs for multi-class classification."""
    rng = np.random.default_rng(seed)
    center_points = rng.uniform(-6, 6, size=(centers, n_features))

    per_center = n_samples // centers
    X_parts, y_parts = [], []
    for c in range(centers):
        n = per_center if c < centers - 1 else n_samples - per_center * (centers - 1)
        X_parts.append(rng.normal(loc=center_points[c], scale=cluster_std, size=(n, n_features)))
        y_parts.append(np.full(n, c))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]
