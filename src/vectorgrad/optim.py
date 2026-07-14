"""
vectorgrad.optim
=================

Optimizers operate directly on `Tensor.data` / `Tensor.grad` NumPy arrays --
no separate "parameter server" abstraction, matching how PyTorch optimizers
walk `param.grad` in place.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        for p, v in zip(self.params, self._velocity):
            if p.grad is None:
                continue
            v *= self.momentum
            v -= self.lr * p.grad
            p.data += v


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014)."""

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
