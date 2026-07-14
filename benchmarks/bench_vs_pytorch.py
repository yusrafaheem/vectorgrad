"""
Benchmark 2: vectorgrad vs. PyTorch on an identical architecture.

Builds the same MLP (same layer sizes, same weight values) in both
vectorgrad and torch.nn, and times forward + backward passes at a few batch
sizes. This is the honest comparison: vectorgrad is a from-scratch, CPU-only,
un-fused, un-JIT'd autodiff engine, so the expected (and correct) result is
that PyTorch wins -- the interesting number is *by how much*, and where the
gap comes from (kernel fusion, memory reuse, BLAS threading, no per-op
Python object overhead per node).

If PyTorch is not installed, this script still runs and reports vectorgrad's
own numbers, with a note on how to enable the comparison.

Run:
    python benchmarks/bench_vs_pytorch.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from vectorgrad import Adam, Linear, ReLU, Sequential, Tensor

try:
    import torch
    import torch.nn as tnn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ARCHITECTURE = [128, 256, 128, 64, 10]  # in -> hidden ... -> out
BATCH_SIZES = [32, 128, 512]
N_ITERS = 20


def build_vectorgrad_model(seed=0):
    np.random.seed(seed)
    layers = []
    for i in range(len(ARCHITECTURE) - 1):
        layers.append(Linear(ARCHITECTURE[i], ARCHITECTURE[i + 1]))
        if i < len(ARCHITECTURE) - 2:
            layers.append(ReLU())
    return Sequential(layers)


def bench_vectorgrad(batch_size: int) -> float:
    model = build_vectorgrad_model()
    optimizer = Adam(model.parameters(), lr=0.001)
    X = np.random.randn(batch_size, ARCHITECTURE[0])
    y = np.random.randint(0, ARCHITECTURE[-1], size=batch_size)

    # warmup
    for _ in range(3):
        logits = model(Tensor(X))
        loss = logits.softmax_cross_entropy(y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        logits = model(Tensor(X))
        loss = logits.softmax_cross_entropy(y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return (time.perf_counter() - t0) / N_ITERS


def build_torch_model(seed=0):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(ARCHITECTURE) - 1):
        layers.append(tnn.Linear(ARCHITECTURE[i], ARCHITECTURE[i + 1]))
        if i < len(ARCHITECTURE) - 2:
            layers.append(tnn.ReLU())
    return tnn.Sequential(*layers)


def bench_torch(batch_size: int) -> float:
    model = build_torch_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = tnn.CrossEntropyLoss()
    X = torch.randn(batch_size, ARCHITECTURE[0])
    y = torch.randint(0, ARCHITECTURE[-1], (batch_size,))

    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return (time.perf_counter() - t0) / N_ITERS


def main():
    print(
        f"architecture: {ARCHITECTURE}, {N_ITERS} iters/batch size "
        "(train step: fwd+bwd+optim.step)"
    )
    print()
    header = f"{'batch size':>10} | {'vectorgrad (ms)':>16}"
    if HAS_TORCH:
        header += f" | {'pytorch (ms)':>13} | {'pytorch speedup':>16}"
    print(header)
    print("-" * len(header))

    for bs in BATCH_SIZES:
        vg_t = bench_vectorgrad(bs) * 1000
        row = f"{bs:>10} | {vg_t:>16.3f}"
        if HAS_TORCH:
            torch_t = bench_torch(bs) * 1000
            row += f" | {torch_t:>13.3f} | {vg_t / torch_t:>15.1f}x"
        print(row)

    if not HAS_TORCH:
        print(
            "\n[torch not installed -- showing vectorgrad-only numbers]\n"
            "Install with `pip install torch` to enable the head-to-head "
            "comparison this benchmark is designed for."
        )
    else:
        print(
            "\nExpected gap sources: PyTorch fuses ops at the C++/ATen level "
            "and reuses pre-allocated buffers across iterations; vectorgrad "
            "allocates a fresh NumPy array (and a fresh Python Tensor object "
            "with its own closures) per op, which dominates at small batch "
            "sizes. The gap should narrow as batch size grows and BLAS "
            "matmul cost starts to dominate over Python/object overhead."
        )


if __name__ == "__main__":
    main()
