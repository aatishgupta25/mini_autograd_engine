import numpy as np

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Sum-reduce `grad` to match `shape`, reversing NumPy broadcasting.
    """
    # 1) Remove extra leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) Sum over dims where original shape was 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    """
    A Tensor wraps a NumPy ndarray and tracks operations for automatic differentiation.
    Supports elementwise ops, matmul, reductions, activations, and backpropagation.
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """Elementwise addition with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            g = out.grad
            self.grad += _unbroadcast(g, self.data.shape)
            other.grad += _unbroadcast(g, other.data.shape)
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        """Elementwise multiplication with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            g = out.grad
            # grad w.r.t. self
            grad_self = other.data * g
            self.grad += _unbroadcast(grad_self, self.data.shape)
            # grad w.r.t. other
            grad_other = self.data * g
            other.grad += _unbroadcast(grad_other, other.data.shape)
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __matmul__(self, other):
        """Matrix multiplication (dot product)."""
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only supports int/float powers"
        out = Tensor(self.data**exponent, (self,), f'**{exponent}')
        def _backward():
            self.grad += exponent * (self.data**(exponent-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """Rectified Linear Unit activation."""
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self,), 'relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward(): self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward(): self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        def _backward(): self.grad += (0.5 / np.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            self.grad += np.broadcast_to(out.grad, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')
        def _backward():
            count = np.prod(self.data.shape if axis is None else np.array(self.data.shape)[axis])
            self.grad += np.broadcast_to(out.grad / count, self.data.shape)
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        data_max = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(data_max, (self,), 'max')
        def _backward():
            mask = (self.data == out.data)
            self.grad += mask * np.broadcast_to(out.grad, self.data.shape)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        sums = np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(exps / sums, (self,), 'softmax')
        def _backward():
            p = out.data
            grad = out.grad
            dot = np.sum(grad * p, axis=axis, keepdims=True)
            self.grad += p * (grad - dot)
        out._backward = _backward
        return out

    def backward(self):
        """Backpropagate to compute gradients for all tensors in the graph."""
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
