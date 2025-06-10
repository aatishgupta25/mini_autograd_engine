import numpy as np

class Value:
    ''' A simple value class to represent a node in the computation graph '''

    def __init__(self, data, _children=(), _op=''):
        self.data = data # The value itself
        self.grad = 0 # Gradient initialized to 0
        self._backward = lambda: None # Default backward function does nothing
        self._prev = set(_children) # Previous values that produced this value
        self._op = _op # Operation that produced this value

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __tanh__(self):
        out = Value(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out
    
    # def __relu__(self):
    #     out = Value(np.maximum(0, self.data), (self,), 'relu')

    #     def _backward():
    #         self.grad += (out.data > 0) * out.grad
    #     out._backward = _backward

    #     return out
    
    # def __sigmoid__(self):
    #     out = Value(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid')

    #     def _backward():
    #         self.grad += out.data * (1 - out.data) * out.grad
    #     out._backward = _backward

    #     return out
    
    def backward(self):
        ''' Backward pass to compute gradients '''
        self.grad = 1.0
        stack = [self]
        while stack:
            v = stack.pop()
            v._backward()
            for child in v._prev:
                if child not in stack:
                    stack.append(child)

    def zero_grad(self):
        ''' Reset gradients to zero '''
        self.grad = 0.0
        for child in self._prev:
            child.zero_grad()

    # def __neg__(self):
    #     out = Value(-self.data, (self,), '-')

    #     def _backward():
    #         self.grad += -out.grad
    #     out._backward = _backward

    #     return out
    # def __sub__(self, other):
    #     out = self + (-other)

    #     def _backward():
    #         self.grad += out.grad
    #         other.grad -= out.grad
    #     out._backward = _backward

    #     return out
    # def __truediv__(self, other):
    #     out = self * other**-1

    #     def _backward():
    #         self.grad += (1 / other.data) * out.grad
    #         other.grad -= (self.data / (other.data ** 2)) * out.grad
    #     out._backward = _backward

    #     return out
    # def __pow__(self, other):
    #     if isinstance(other, Value):
    #         out = Value(self.data ** other.data, (self, other), '**')
    #         def _backward():
    #             self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
    #             other.grad += (out.data * np.log(self.data)) * out.grad
    #         out._backward = _backward
    #     else:
    #         out = Value(self.data ** other, (self,), '**')
    #         def _backward():
    #             self.grad += (other * (self.data ** (other - 1))) * out.grad
    #         out._backward = _backward

    #     return out
    # def __radd__(self, other):
    #     return self + other
    # def __rmul__(self, other):
    #     return self * other
    # def __rsub__(self, other):
    #     return Value(other) - self
    # def __rtruediv__(self, other):
    #     return Value(other) / self
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"