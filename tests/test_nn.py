import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from vectorgrad import Linear, ReLU, Sequential, Tensor


class TestLinear(unittest.TestCase):
    def test_output_shape(self):
        layer = Linear(10, 4)
        x = Tensor(np.random.randn(8, 10))
        out = layer(x)
        self.assertEqual(out.shape, (8, 4))

    def test_parameter_shapes(self):
        layer = Linear(10, 4)
        params = layer.parameters()
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, (10, 4))  # weight
        self.assertEqual(params[1].shape, (4,))  # bias

    def test_he_init_scale_roughly_correct(self):
        # He init: std ~= sqrt(2 / fan_in). Not a tight bound, just a sanity check.
        layer = Linear(1000, 50, init="he")
        empirical_std = layer.weight.data.std()
        expected_std = np.sqrt(2.0 / 1000)
        self.assertAlmostEqual(empirical_std, expected_std, delta=expected_std * 0.3)

    def test_unknown_init_raises(self):
        with self.assertRaises(ValueError):
            Linear(4, 4, init="bogus")


class TestSequential(unittest.TestCase):
    def setUp(self):
        self.model = Sequential([Linear(6, 12), ReLU(), Linear(12, 3)])

    def test_forward_shape(self):
        x = Tensor(np.random.randn(5, 6))
        out = self.model(x)
        self.assertEqual(out.shape, (5, 3))

    def test_parameter_count(self):
        # (6*12 + 12) + (12*3 + 3) = 84 + 39 = 123
        n_params = sum(p.data.size for p in self.model.parameters())
        self.assertEqual(n_params, 123)

    def test_relu_zeroes_negative_activations(self):
        x = Tensor(np.array([[-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]]))
        first_linear = self.model.layers[0]
        pre_activation = first_linear(x)
        post_activation = self.model.layers[1](pre_activation)
        self.assertTrue((post_activation.data >= 0).all())


if __name__ == "__main__":
    unittest.main()
