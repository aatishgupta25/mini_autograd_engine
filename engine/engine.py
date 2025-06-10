
class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    