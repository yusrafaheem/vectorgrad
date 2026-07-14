"""
Gradient checking: for every op, compare the analytic gradient produced by
Tensor.backward() against a central-difference numerical approximation.

This is the same technique Andrew Ng's course (and the source tutorial this
project grew out of) recommends, and it is the single most important test
category in an autodiff engine -- a bug in a backward() closure will not
raise an exception, it will silently produce wrong gradients.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from vectorgrad.tensor import Tensor

np.random.seed(0)


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference numerical gradient of scalar-valued f w.r.t. x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        plus = f(x)

        x[idx] = orig - eps
        minus = f(x)

        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


class GradCheckMixin:
    def assert_grad_matches(self, forward_fn, x: np.ndarray, atol=1e-4, rtol=1e-3):
        """forward_fn(np.ndarray) -> Tensor (scalar output)."""
        x = x.astype(np.float64).copy()

        t = Tensor(x.copy(), requires_grad=True)
        out = forward_fn(t)
        out.backward()
        analytic = t.grad

        def f(arr):
            return float(forward_fn(Tensor(arr)).data)

        numeric = numerical_grad(f, x.copy())

        np.testing.assert_allclose(analytic, numeric, atol=atol, rtol=rtol)


class TestElementaryOps(unittest.TestCase, GradCheckMixin):
    def test_add(self):
        self.assert_grad_matches(lambda t: (t + 3.0).sum(), np.random.randn(5))

    def test_sub(self):
        self.assert_grad_matches(lambda t: (t - 2.0).sum(), np.random.randn(5))

    def test_mul(self):
        self.assert_grad_matches(lambda t: (t * t).sum(), np.random.randn(5))

    def test_pow(self):
        self.assert_grad_matches(lambda t: (t ** 3).sum(), np.random.randn(5) + 2.0)

    def test_relu(self):
        self.assert_grad_matches(lambda t: t.relu().sum(), np.random.randn(20))

    def test_sigmoid(self):
        self.assert_grad_matches(lambda t: t.sigmoid().sum(), np.random.randn(10))

    def test_tanh(self):
        self.assert_grad_matches(lambda t: t.tanh().sum(), np.random.randn(10))

    def test_mean(self):
        self.assert_grad_matches(lambda t: t.mean(), np.random.randn(7))


class TestMatmulAndBroadcast(unittest.TestCase, GradCheckMixin):
    def test_matmul(self):
        W = np.random.randn(4, 3)

        def forward(t):
            w = Tensor(W)
            return t.matmul(w).sum()

        self.assert_grad_matches(forward, np.random.randn(5, 4))

    def test_broadcast_add_bias(self):
        b = np.random.randn(3)

        def forward(t):
            bias = Tensor(b)
            return (t + bias).sum()

        self.assert_grad_matches(forward, np.random.randn(6, 3))

    def test_linear_layer_composition(self):
        """A full Linear-like forward: x @ W + b, then ReLU, then sum."""
        W = np.random.randn(4, 5)
        b = np.random.randn(5)

        def forward(t):
            return (t.matmul(Tensor(W)) + Tensor(b)).relu().sum()

        self.assert_grad_matches(forward, np.random.randn(3, 4))


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def test_gradient_matches_numerical(self):
        logits = np.random.randn(6, 4)
        targets = np.array([0, 1, 2, 3, 1, 0])

        t = Tensor(logits.copy(), requires_grad=True)
        loss = t.softmax_cross_entropy(targets)
        loss.backward()
        analytic = t.grad

        def f(arr):
            tt = Tensor(arr)
            return float(tt.softmax_cross_entropy(targets).data)

        numeric = numerical_grad(f, logits.copy())
        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)

    def test_loss_decreases_with_perfect_logits(self):
        # A very confident, correct prediction should have near-zero loss.
        logits = np.array([[50.0, -50.0, -50.0]])
        targets = np.array([0])
        loss = Tensor(logits).softmax_cross_entropy(targets)
        self.assertLess(float(loss.data), 1e-6)

    def test_no_nan_on_large_logits(self):
        # Regression test for the overflow bug the source tutorial called out.
        logits = np.array([[1000.0, 1.0, -1000.0]])
        targets = np.array([0])
        loss = Tensor(logits).softmax_cross_entropy(targets)
        self.assertFalse(np.isnan(loss.data))


class TestBackwardMechanics(unittest.TestCase):
    def test_gradients_accumulate_not_overwrite(self):
        """Using the same tensor twice in a graph must sum gradients."""
        x = Tensor(np.array([2.0]), requires_grad=True)
        y = x + x  # dy/dx should be 2
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([2.0]))

    def test_zero_grad_resets(self):
        x = Tensor(np.array([2.0]), requires_grad=True)
        (x * x).backward()
        self.assertIsNotNone(x.grad)
        x.zero_grad()
        self.assertIsNone(x.grad)


if __name__ == "__main__":
    unittest.main()
