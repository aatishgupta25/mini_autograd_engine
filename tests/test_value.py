import numpy as np
import torch
from engine.scalar_engine import Value

def test_gradients_through_chain():
    x = Value(0.1)
    y = Value(0.2)
    z = Value(0.3)
    expr = ((x + y) * z).tanh()
    expr.backward()
    assert x.grad != 0
    assert y.grad != 0
    assert z.grad != 0

def test_tanh_near_zero():
    a = Value(1e-8)
    out = a.tanh()
    out.backward()
    assert abs(out.data - 1e-8) < 1e-9
    assert abs(a.grad - 1.0) < 1e-6

def test_power_zero_exponent():
    a = Value(5.0)
    out = a ** 0
    out.backward()
    assert out.data == 1.0
    assert a.grad == 0.0

def test_power_negative():
    a = Value(2.0)
    out = a ** -2
    out.backward()
    assert abs(out.data - 0.25) < 1e-6
    expected_grad = -2 * (2.0 ** -3)  # -0.25
    assert abs(a.grad - expected_grad) < 1e-6

def test_division_gradient_symmetry():
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

    assert grad_a1 != grad_a2
    assert grad_b1 != grad_b2

def test_gradient_accumulation():
    a = Value(2.0)
    b = Value(3.0)
    c1 = a * b
    c2 = a + b
    d = c1 + c2
    d.backward()
    assert abs(a.grad - (b.data + 1)) < 1e-6
    assert abs(b.grad - (a.data + 1)) < 1e-6

def test_tiny_inputs_chain_stability():
    x = Value(1e-6)
    y = Value(1e-6)
    z = (x * y + x / y).tanh()
    z.backward()
    assert np.isfinite(x.grad)
    assert np.isfinite(y.grad)

def test_backward_topology_order():
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    d = a * b
    e = c * d
    f = e.tanh()
    f.backward()
    assert a.grad > 0
    assert b.grad > 0
    assert c.grad > 0
    assert d.grad > 0
    assert e.grad > 0

def test_deep_expression_tree():
    x = Value(0.5)
    for _ in range(100):
        x = x * 1.01 + 0.001
    x.backward()
    assert np.isfinite(x.grad)

def test_zero_gradient_sink():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y
    w = z * 0
    w.backward()
    assert x.grad == 0.0
    assert y.grad == 0.0

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert abs(ymg.data - ypt.item()) < 1e-6
    assert abs(xmg.grad - xpt.grad.item()) < 1e-6

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
    b = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    assert abs(gmg.data - gpt.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
