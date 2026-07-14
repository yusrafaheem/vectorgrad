import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from vectorgrad import SGD, Adam, Tensor


def quadratic_bowl_step(optimizer_cls, steps=200, **kwargs):
    """Minimize f(x) = sum((x - target)^2) and check convergence.

    A well-implemented first-order optimizer should drive x to `target`
    on this convex, well-conditioned problem regardless of starting point.
    """
    target = np.array([3.0, -2.0, 5.0])
    x = Tensor(np.zeros(3), requires_grad=True)
    optimizer = optimizer_cls([x], **kwargs)

    for _ in range(steps):
        diff = x - Tensor(target)
        loss = (diff * diff).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return x.data, target


def quadratic_bowl_error_trajectory(optimizer_cls, steps, **kwargs):
    """Like quadratic_bowl_step, but returns the L2 error at every step."""
    target = np.array([3.0, -2.0, 5.0])
    x = Tensor(np.zeros(3), requires_grad=True)
    optimizer = optimizer_cls([x], **kwargs)

    errors = []
    for _ in range(steps):
        diff = x - Tensor(target)
        loss = (diff * diff).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(float(np.linalg.norm(x.data - target)))
    return errors


class TestSGD(unittest.TestCase):
    def test_converges_on_convex_problem(self):
        x_final, target = quadratic_bowl_step(SGD, steps=300, lr=0.05)
        np.testing.assert_allclose(x_final, target, atol=1e-2)

    def test_momentum_speeds_up_convergence(self):
        # Compare how many steps each variant needs to first cross a fixed
        # error threshold, rather than comparing the error at a fixed step
        # count -- momentum trajectories oscillate around the minimum, so a
        # snapshot comparison is flaky depending on where that snapshot
        # lands in the oscillation cycle.
        threshold = 0.5
        steps = 60

        plain_errors = quadratic_bowl_error_trajectory(SGD, steps, lr=0.02, momentum=0.0)
        momentum_errors = quadratic_bowl_error_trajectory(SGD, steps, lr=0.02, momentum=0.9)

        def first_below(errors, thresh):
            for i, e in enumerate(errors):
                if e < thresh:
                    return i
            return len(errors)  # never converged within budget

        momentum_steps = first_below(momentum_errors, threshold)
        plain_steps = first_below(plain_errors, threshold)
        self.assertLess(momentum_steps, plain_steps)


class TestAdam(unittest.TestCase):
    def test_converges_on_convex_problem(self):
        x_final, target = quadratic_bowl_step(Adam, steps=300, lr=0.1)
        np.testing.assert_allclose(x_final, target, atol=1e-2)

    def test_zero_grad_prevents_stale_updates(self):
        x = Tensor(np.array([1.0]), requires_grad=True)
        optimizer = Adam([x], lr=0.1)

        (x * x).backward()
        optimizer.zero_grad()
        # No backward() called after zero_grad -> grad is None -> step() must no-op.
        before = x.data.copy()
        optimizer.step()
        np.testing.assert_allclose(x.data, before)


if __name__ == "__main__":
    unittest.main()
