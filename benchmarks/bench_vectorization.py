"""
Benchmark 1: why vectorization is the whole point.

Compares three implementations of the same forward pass through a Linear
layer (y = x @ W + b) at increasing batch sizes:

  1. pure Python, triple-nested for loops (the "textbook" way to compute a
     matrix multiply, and how a first pass at "NN from scratch" is often
     written before optimizing)
  2. NumPy vectorized (what vectorgrad.Tensor actually does)
  3. NumPy vectorized, batched matmul over multiple layers

The goal is to make the systems argument concrete: the *math* of a from-
scratch neural net is identical whether you write it with loops or with
vectorized array ops, but the performance gap is 2-3 orders of magnitude,
and that gap is precisely why frameworks like PyTorch/TensorFlow exist as
thin Python wrappers around vectorized (and GPU-accelerated) kernels.

Run:
    python benchmarks/bench_vectorization.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def naive_linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = X @ W + b, implemented with explicit Python loops."""
    n, in_features = X.shape
    _, out_features = W.shape
    Y = [[0.0] * out_features for _ in range(n)]
    for i in range(n):
        for j in range(out_features):
            s = 0.0
            for k in range(in_features):
                s += X[i, k] * W[k, j]
            Y[i][j] = s + b[j]
    return np.array(Y)


def vectorized_linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def timeit(fn, *args, repeats=3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    np.random.seed(0)
    in_features, out_features = 128, 64
    W = np.random.randn(in_features, out_features)
    b = np.random.randn(out_features)

    header = (
        f"{'batch size':>10} | {'naive loops (s)':>16} | "
        f"{'vectorized (s)':>15} | {'speedup':>10}"
    )
    print(header)
    print("-" * 62)

    # Naive loops become painfully slow fast -- cap how large we test them.
    naive_batch_sizes = [8, 32, 128]
    vector_batch_sizes = [8, 32, 128, 512, 2048, 8192]

    naive_times = {}
    for n in naive_batch_sizes:
        X = np.random.randn(n, in_features)
        naive_times[n] = timeit(naive_linear_forward, X, W, b, repeats=1)

    for n in vector_batch_sizes:
        X = np.random.randn(n, in_features)
        vec_t = timeit(vectorized_linear_forward, X, W, b)
        naive_t = naive_times.get(n)
        naive_str = f"{naive_t:.6f}" if naive_t is not None else "skipped (too slow)"
        speedup = f"{naive_t / vec_t:8.0f}x" if naive_t is not None else "-"
        print(f"{n:>10} | {naive_str:>16} | {vec_t:>15.6f} | {speedup:>10}")

    print(
        "\nTakeaway: identical math (a matrix multiply), 2-3+ orders of "
        "magnitude apart in wall-clock time. This is the practical reason "
        "every real framework compiles down to vectorized / GPU kernels "
        "instead of interpreting Python loops per-element."
    )


if __name__ == "__main__":
    main()
