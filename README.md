# vectorgrad

A reverse-mode automatic differentiation engine and neural network library, built from scratch on top of vectorized NumPy — no PyTorch, no TensorFlow, no autograd dependency. Includes a benchmark suite comparing it against PyTorch to make the systems tradeoffs concrete rather than theoretical.

## Why this exists

Most "neural network from scratch" projects stop at "does it train MNIST." That's a good first milestone, but it doesn't say much about the actual engineering problem deep learning frameworks solve: how do you make automatic differentiation *fast*? This project treats that as the central question.

Concretely, `vectorgrad` is built around three ideas:

1. **A real (if minimal) autodiff engine, not a hand-derived backprop script.** `Tensor` builds a dynamic computational graph (define-by-run, like PyTorch), and gradients are computed generically by walking that graph in reverse topological order — the same mechanism production autograd systems use, just without the compiler.
2. **Every op is vectorized.** Gradients are computed for whole batches via NumPy array ops, not per-element Python loops. `benchmarks/bench_vectorization.py` quantifies exactly how much this matters (spoiler: 3+ orders of magnitude).
3. **Honest benchmarking against PyTorch**, not just "it works." `benchmarks/bench_vs_pytorch.py` runs an identical architecture through both `vectorgrad` and `torch.nn`, and reports where and why the gap exists (kernel fusion, buffer reuse, per-op Python/object overhead).

## Project structure

```
vectorgrad/
├── src/vectorgrad/
│   ├── tensor.py     # autodiff engine: Tensor, backward(), all ops
│   ├── nn.py          # Module, Linear, ReLU/Sigmoid/Tanh, Sequential
│   ├── optim.py        # SGD (with momentum), Adam
│   └── data.py          # dependency-free synthetic data + batch iterator
├── tests/            # unit tests, incl. finite-difference gradient checking
├── benchmarks/         # vectorization + PyTorch comparison benchmarks
├── examples/            # end-to-end training script
└── .github/workflows/  # CI: lint, format check, tests, smoke-run examples
```

## Quickstart

```bash
git clone https://github.com/yusrafaheem/vectorgrad.git
cd vectorgrad
pip install -e ".[examples]"

python examples/train_classifier.py
```

```python
import numpy as np
from vectorgrad import Tensor, Linear, ReLU, Sequential, Adam

model = Sequential([
    Linear(8, 64), ReLU(),
    Linear(64, 32), ReLU(),
    Linear(32, 6),
])
optimizer = Adam(model.parameters(), lr=0.01)

x = Tensor(np.random.randn(32, 8))
labels = np.random.randint(0, 6, size=32)

logits = model(x)
loss = logits.softmax_cross_entropy(labels)

model.zero_grad()
loss.backward()
optimizer.step()
```

## Testing

Every op in `tensor.py` is checked against a central-difference numerical gradient, not just spot-tested — this is the standard technique for catching silently-wrong backward passes (the failure mode where nothing crashes, but training just doesn't work). Also includes shape/composition tests for `nn.py` and convergence tests for both optimizers on a convex problem.

```bash
python -m unittest discover -s tests -v
# Ran 27 tests in 0.06s — OK
```

CI (`.github/workflows/ci.yml`) runs this matrix across Python 3.9–3.12, plus `ruff` lint, `black --check` formatting, and a smoke-run of the example and benchmark scripts, on every push and PR.

## Benchmarks

### 1. Why vectorization matters

`benchmarks/bench_vectorization.py` times the identical `y = X @ W + b` computation implemented two ways: naive triple-nested Python loops vs. vectorized NumPy. Measured run (128→64 Linear layer):

```
batch size | naive loops (s) | vectorized (s) | speedup
------------------------------------------------------
         8 |        0.010541 |       0.000009 |   1119x
        32 |        0.041657 |       0.000021 |   2000x
       128 |        0.173729 |       0.000032 |   5429x
       512 |       (skipped) |       0.000083 |       -
      2048 |       (skipped) |       0.000644 |       -
      8192 |       (skipped) |       0.007115 |       -
```

The naive version was skipped past batch size 128 because it becomes impractically slow. This is the concrete case for why every real framework compiles down to vectorized (and eventually GPU) kernels instead of interpreting Python loops per scalar.

### 2. vectorgrad vs. PyTorch

`benchmarks/bench_vs_pytorch.py` builds the same architecture (`128 → 256 → 128 → 64 → 10`, ReLU, Adam) in both libraries and times a full train step (forward + backward + optimizer step). If PyTorch isn't installed, the script still runs and reports vectorgrad's own numbers with a note on enabling the comparison — install `torch` and re-run to see the head-to-head:

```bash
pip install torch   # CPU build is enough
python benchmarks/bench_vs_pytorch.py
```

The expected result is that PyTorch wins, and by a wide margin at small batch sizes — the interesting part is *why*: PyTorch fuses operations at the ATen/C++ level and reuses pre-allocated buffers across iterations, while vectorgrad allocates a fresh NumPy array and a fresh Python `Tensor` (with its own backward closure) per op. That gap should narrow as batch size grows and matmul FLOPs start to dominate over per-op Python overhead — which is itself a useful, measurable claim about where the bottleneck lives.

## Roadmap

- [ ] Batch normalization and dropout layers
- [ ] A `no_grad()` context manager for inference-only forward passes
- [ ] Simple computational-graph visualization (export to Graphviz)
- [ ] Optional Numba/Cython-accelerated ops to close part of the PyTorch gap and measure how much
- [ ] Extend `benchmarks/` to profile memory allocation, not just wall-clock time

## Background

This project started from working through [*How to build a neural network from zero*](https://towardsdatascience.com/building-a-neural-network-from-scratch-8f03c5c50adc/) (Bernardino Sassoli, *Towards Data Science*), which implements a single-hidden-layer MLP on MNIST with hand-derived forward/backward passes. `vectorgrad` takes that as a starting point but generalizes it into a real autodiff engine (arbitrary graphs, not one hard-coded architecture) and adds the testing, benchmarking, and systems-performance angle that a from-scratch tutorial doesn't cover.

## License

MIT — see [LICENSE](LICENSE).
