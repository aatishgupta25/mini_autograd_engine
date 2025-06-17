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
        other = other if isinstance(other, Value) else Value(other) # Ensure other is a Value
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad # cannot be just = out.grad because we want to accumulate gradients
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other): # A reflection method for addition
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Ensure other is a Value
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): # A reflection method for multiplication
        return self * other
    
    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):

        ''' Topological order all of the children in the graph - starting from the 
        very first node to building up to the final output node this is done by 
        recursively visiting each node and adding it to a list '''

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        out = Value(-self.data, (self,), '-')

        def _backward():
            self.grad += -out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        if other == 0:
            raise ZeroDivisionError("Division by zero in Value.__truediv__")
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        if self.data == 0:
            raise ZeroDivisionError("Division by zero in Value.__rtruediv__")
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
def with_pytorch():

    import torch

    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    # Operations
    a = x * 2
    b = y + z
    c = a - b
    d = c ** 2
    e = d / 3
    f = torch.tanh(e)

    # Backward pass
    f.backward()

    print("From PyTorch API: x.grad =", x.grad.item())
    print("From PyTorch API: y.grad =", y.grad.item())
    print("From PyTorch API: z.grad =", z.grad.item())

if __name__ == "__main__":
    from make_graph_for_viz import draw_dot # Putting this import here to avoid circular imports
    
    x = Value(1.0)
    y = Value(2.0)
    z = Value(3.0)

    a = x * 2
    b = y + z
    c = a - b
    d = c ** 2
    e = d / 3
    f = e.tanh()

    draw_dot(f, filename='graph_before_backprop')

    # Backward pass
    f.backward()

    draw_dot(f, filename='graph_after_backprop')

    print("From our engine: x.grad =", x.grad)
    print("From our engine: y.grad =", y.grad)
    print("From our engine: z.grad =", z.grad)

    with_pytorch()
