import numpy as np
import pytest
from engine.tensor_engine import Tensor
from engine.tensor_optimizer import SGD, Adam

@pytest.fixture
def setup_vars():
    w = Tensor(2.0)
    b = Tensor(0.5)
    x = Tensor(3.0)
    target = 10.0
    return w, b, x, target

def test_sgd_updates_params(setup_vars):
    w, b, x, target = setup_vars
    opt = SGD([w, b], lr=0.1, momentum=0.0)

    # Forward and backward
    y = w * x + b
    loss = (y - target) ** 2
    loss.backward()

    w_before, b_before = w.data.copy(), b.data.copy()
    assert not np.isclose(w.grad, 0.0)
    assert not np.isclose(b.grad, 0.0)

    opt.step()

    # Check that params have been updated
    assert not np.isclose(w.data, w_before)
    assert not np.isclose(b.data, b_before)

def test_adam_param_changes_across_steps(setup_vars):
    w, b, x, target = setup_vars
    opt = Adam([w, b], lr=0.1)

    values = []
    for _ in range(2):
        y = w * x + b
        loss = (y - target) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        values.append((w.data.item(), b.data.item()))

    # Check that both w and b changed across the two steps
    w0, b0 = values[0]
    w1, b1 = values[1]
    assert not np.isclose(w0, w1)
    assert not np.isclose(b0, b1)
