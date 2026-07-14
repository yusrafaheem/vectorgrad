"""
vectorgrad.nn
=============

Small `Module` / `Linear` / `Sequential` API, deliberately shaped like
PyTorch's `nn.Module` so the parallels (and the benchmark comparison) are
direct. Everything is built on top of `vectorgrad.tensor.Tensor`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .tensor import Tensor


class Module:
    """Base class for anything with learnable parameters."""

    def parameters(self) -> List[Tensor]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """Fully-connected layer: y = x @ W + b.

    Weight initialization defaults to He initialization (good pairing with
    ReLU); pass `init="xavier"` for tanh/sigmoid networks.
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he"):
        self.in_features = in_features
        self.out_features = out_features

        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown init scheme: {init}")

        w = np.random.randn(in_features, out_features) * scale
        b = np.zeros(out_features)

        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features})"


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sequential(Module):
    """Chains a list of modules, e.g.

    model = Sequential([
        Linear(784, 128),
        ReLU(),
        Linear(128, 32),
        ReLU(),
        Linear(32, 10),
    ])
    """

    def __init__(self, layers: List[Module]):
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        body = "\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {body}\n)"
