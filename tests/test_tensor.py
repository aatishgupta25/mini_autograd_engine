import numpy as np
from engine.tensor_engine import Tensor

def test_scalar_operations():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b + a  # 2 * 3 + 2 = 8
    c.backward()
    assert np.allclose(a.grad, 4.0)  # b + 1
    assert np.allclose(b.grad, 2.0)  # a

def test_vector_sum():
    v = Tensor([1.0, 2.0, 3.0])
    s = v.sum()
    s.backward()
    assert np.allclose(v.grad, [1.0, 1.0, 1.0])

def test_matmul_and_sum():
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = A @ B
    D = C.sum()
    D.backward()

    expected_A_grad = np.array([[1.0, 1.0, 2.0],
                                [1.0, 1.0, 2.0]])
    expected_B_grad = np.array([[5.0, 5.0],
                                [7.0, 7.0],
                                [9.0, 9.0]])
    assert np.allclose(A.grad, expected_A_grad)
    assert np.allclose(B.grad, expected_B_grad)

def test_exp_log_identity():
    x = Tensor(1.0)
    y = x.exp().log()  # log(exp(x)) == x
    y.backward()
    assert np.allclose(x.grad, 1.0)

def test_sqrt_pow_identity():
    a = Tensor(4.0)
    b = a.sqrt() ** 2  # (sqrt(a))^2 == a
    b.backward()
    assert np.allclose(a.grad, 1.0)

def test_max_gradient():
    v = Tensor([1.0, 3.0, 2.0])
    m = v.max()
    m.backward()
    assert np.allclose(v.grad, [0.0, 1.0, 0.0])

def test_softmax_gradient_sum_zero():
    logits = Tensor([2.0, 1.0, 0.1])
    probs = logits.softmax()
    loss = probs.sum()
    loss.backward()
    assert np.allclose(logits.grad.sum(), 0.0)
