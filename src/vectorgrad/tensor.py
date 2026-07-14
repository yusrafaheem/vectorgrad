"""
vectorgrad.tensor
==================

A minimal reverse-mode automatic differentiation engine, implemented directly
on top of vectorized NumPy arrays (as opposed to the scalar-graph approach
used by teaching engines like micrograd).

Design goals (this is the "systems" angle of the project, not just a
from-scratch NN tutorial):

1. Every ``Tensor`` operation is a *vectorized* NumPy op -- gradients are
   computed for whole arrays/batches at once, the same way production
   frameworks operate, rather than element-by-element.
2. The computational graph is built dynamically (define-by-run, like
   PyTorch's autograd) via closures stored on each node.
3. Broadcasting is handled explicitly in every backward closure by summing
   gradients back down to the original (pre-broadcast) shape -- this is the
   single most common source of bugs when implementing autodiff by hand, so
   it is centralized in :func:`_unbroadcast`.
4. A lightweight op-level profiler hook is included (see ``Tensor.op_stats``)
   so forward/backward cost can be measured per operation type -- useful for
   the benchmark suite in ``benchmarks/``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np

# Global, resettable op-timing registry used by the benchmark suite.
OP_STATS: dict[str, list[float]] = defaultdict(list)


def reset_op_stats() -> None:
    OP_STATS.clear()


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum-reduce `grad` back down to `shape`, undoing NumPy broadcasting.

    If a tensor of shape (3,) was broadcast against one of shape (32, 3)
    during a forward op, the incoming gradient will have shape (32, 3) and
    must be summed over axis 0 to match the original (3,) parameter.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A NumPy-backed node in the autodiff graph.

    Parameters
    ----------
    data:
        The underlying array (converted to float64 NumPy array).
    requires_grad:
        Whether gradients should be accumulated for this tensor.
    _children:
        Internal: parent tensors in the graph (do not set manually).
    _op:
        Internal: label of the op that produced this tensor (for
        introspection/debugging and the profiler).
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op", "probs")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple = (),
        _op: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'})"

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------ #
    # Helper to build a result tensor + record timing for the benchmark
    # ------------------------------------------------------------------ #
    @staticmethod
    def _time_op(op_name, fn):
        t0 = time.perf_counter()
        out = fn()
        OP_STATS[op_name].append(time.perf_counter() - t0)
        return out

    # ------------------------------------------------------------------ #
    # Elementary ops
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, req, (self, other), "add")

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, req, (self, other), "mul")

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad * other.data, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad * self.data, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, power: float):
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"pow{power}")

        def _backward():
            if self.requires_grad:
                g = (power * self.data ** (power - 1)) * out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        assert isinstance(other, Tensor)
        req = self.requires_grad or other.requires_grad

        def _fwd():
            return self.data @ other.data

        data = self._time_op("matmul_fwd", _fwd)
        out = Tensor(data, req, (self, other), "matmul")

        def _backward():
            def _bwd():
                if self.requires_grad:
                    g = out.grad @ other.data.swapaxes(-1, -2)
                    self.grad = g if self.grad is None else self.grad + g
                if other.requires_grad:
                    g = self.data.swapaxes(-1, -2) @ out.grad
                    other.grad = g if other.grad is None else other.grad + g

            self._time_op("matmul_bwd", _bwd)

        out._backward = _backward
        return out

    __matmul__ = matmul

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            (self,),
            "sum",
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.data.shape).astype(np.float64).copy()
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------ #
    # Activations
    # ------------------------------------------------------------------ #
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                g = (self.data > 0).astype(np.float64) * out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                g = (s * (1 - s)) * out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                g = (1 - t ** 2) * out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Loss: numerically-stable softmax + cross-entropy fused op
    # ------------------------------------------------------------------ #
    def softmax_cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """Fused softmax + NLL loss.

        `self` is the (batch, num_classes) logits tensor. `targets` is an
        integer array of shape (batch,) with the correct class indices.

        Fusing softmax and cross-entropy (rather than composing separate
        softmax() and log() ops) avoids computing exp() twice and gives a
        much simpler, numerically-stable gradient: (softmax(logits) - one_hot).
        This mirrors how production frameworks implement
        `CrossEntropyLoss` for both numerical stability and speed.
        """
        targets = np.asarray(targets)
        batch = self.data.shape[0]

        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        log_likelihood = -np.log(np.clip(probs[np.arange(batch), targets], 1e-12, None))
        loss_val = log_likelihood.mean()

        out = Tensor(loss_val, self.requires_grad, (self,), "softmax_xent")

        def _backward():
            if self.requires_grad:
                grad = probs.copy()
                grad[np.arange(batch), targets] -= 1
                grad /= batch
                grad = grad * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        out.probs = probs  # stashed for accuracy computation without recompute
        return out

    # ------------------------------------------------------------------ #
    # Shape ops
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                g = out.grad.reshape(orig_shape)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        axes = axes or None
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                if axes is None:
                    g = out.grad.transpose()
                else:
                    inv = np.argsort(axes)
                    g = out.grad.transpose(inv)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Backward pass: reverse topological order, exactly once per node
    # ------------------------------------------------------------------ #
    def backward(self):
        topo: list[Tensor] = []
        visited = set()

        def build(v: "Tensor"):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
