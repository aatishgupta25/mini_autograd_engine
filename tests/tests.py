import unittest
import numpy as np
from engine.engine import Value

class Test_Engine(unittest.TestCase):

    def test_gradients_through_chain(self):
        x = Value(0.1)
        y = Value(0.2)
        z = Value(0.3)
        expr = ((x + y) * z).tanh()
        expr.backward()
        self.assertNotEqual(x.grad, 0)
        self.assertNotEqual(y.grad, 0)
        self.assertNotEqual(z.grad, 0)

    def test_tanh_near_zero(self):
        a = Value(1e-8)
        out = a.tanh()
        out.backward()
        self.assertAlmostEqual(out.data, 1e-8, delta=1e-9)
        self.assertAlmostEqual(a.grad, 1.0, delta=1e-6)

    def test_power_zero_exponent(self):
        a = Value(5.0)
        out = a ** 0
        out.backward()
        self.assertAlmostEqual(out.data, 1.0)
        self.assertEqual(a.grad, 0.0)

    def test_power_negative(self):
        a = Value(2.0)
        out = a ** -2
        out.backward()
        self.assertAlmostEqual(out.data, 0.25)
        expected_grad = -2 * (2.0 ** -3)  # -2 * 0.125 = -0.25
        self.assertAlmostEqual(a.grad, expected_grad)

    def test_division_gradient_symmetry(self):
        a = Value(4.0)
        b = Value(2.0)
        out1 = a / b
        out1.backward()
        grad_a1, grad_b1 = a.grad, b.grad

        a = Value(4.0)
        b = Value(2.0)
        out2 = b / a
        out2.backward()
        grad_b2, grad_a2 = b.grad, a.grad

        self.assertNotAlmostEqual(grad_a1, grad_a2)
        self.assertNotAlmostEqual(grad_b1, grad_b2)

    def test_gradient_accumulation(self):
        a = Value(2.0)
        b = Value(3.0)
        c1 = a * b
        c2 = a + b
        d = c1 + c2
        d.backward()
        # a appears in both c1 and c2 — grad should accumulate
        expected_grad_a = b.data + 1  # from a*b (grad = b), from a+b (grad = 1)
        expected_grad_b = a.data + 1
        self.assertAlmostEqual(a.grad, expected_grad_a)
        self.assertAlmostEqual(b.grad, expected_grad_b)

    def test_tiny_inputs_chain_stability(self):
        x = Value(1e-6)
        y = Value(1e-6)
        z = (x * y + x / y).tanh()
        z.backward()
        self.assertTrue(np.isfinite(x.grad))
        self.assertTrue(np.isfinite(y.grad))

    def test_backward_topology_order(self):
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        d = a * b
        e = c * d
        f = e.tanh()
        f.backward()
        self.assertGreater(a.grad, 0)
        self.assertGreater(b.grad, 0)
        self.assertGreater(c.grad, 0)
        self.assertGreater(d.grad, 0)
        self.assertGreater(e.grad, 0)

    def test_deep_expression_tree(self):
        x = Value(0.5)
        for _ in range(100):
            x = x * 1.01 + 0.001  # simulate parameter growth
        x.backward()
        self.assertTrue(np.isfinite(x.grad))

    def test_zero_gradient_sink(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        w = z * 0  # gradient sink — should zero out
        w.backward()
        self.assertEqual(x.grad, 0.0)
        self.assertEqual(y.grad, 0.0)


if __name__ == "__main__":
    unittest.main()
