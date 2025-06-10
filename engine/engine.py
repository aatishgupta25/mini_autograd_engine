
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
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"