from engine.tensor_engine import Tensor
import numpy as np

class Module:
    """Base class for models."""
    def zero_grad(self):
        for p in self.parameters(): p.grad = np.zeros_like(p.data)
    def parameters(self): return []

class Neuron(Module):
    """Single neuron with optional ReLU."""
    def __init__(self, nin, nonlin=True):
        self.w = [Tensor(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Tensor(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self): return self.w + [self.b]
    def __repr__(self): return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """Layer of multiple neurons."""
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    def __repr__(self): return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """Multi-layer perceptron that outputs a single vector Tensor."""
    def __init__(self, nin, nouts):
        sz = [nin] + list(nouts)
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1))
            for i in range(len(nouts))
        ]

    def __call__(self, x_list):
        # x_list: list of Tensor scalars
        out = x_list
        for layer in self.layers:
            out = layer(out)  # may be list of Tensors
        # If final output is a list, stack into one Tensor
        if isinstance(out, list) and len(out) > 1:
            # Stack data arrays
            data = np.stack([t.data for t in out], axis=-1)  # shape (n,)
            vec = Tensor(data, tuple(out), 'stack')

            def _backward():
                # vec.grad is shape (n,)
                for i, t in enumerate(out):
                    t.grad += vec.grad[..., i]  # send slice back
            vec._backward = _backward

            return vec

        # If single output, just return it
        return out if isinstance(out, Tensor) else out[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def __repr__(self): return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    def __call__(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.data.shape) > self.p)
            return Tensor(x.data * self.mask, (x,), 'dropout')
        else:
            return Tensor(x.data * (1 - self.p), (x,), 'dropout')
    def parameters(self): return []

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma = Tensor(np.ones(dim))
        self.beta  = Tensor(np.zeros(dim))
        self.eps, self.mom = eps, momentum
        self.running_mean = np.zeros(dim)
        self.running_var  = np.ones(dim)
    def __call__(self, x):
        if self.training:
            mu = np.mean(x.data, axis=0)
            var = np.var(x.data, axis=0)
            self.running_mean = self.mom*self.running_mean + (1-self.mom)*mu
            self.running_var  = self.mom*self.running_var  + (1-self.mom)*var
            x_hat = (x.data - mu) / np.sqrt(var + self.eps)
        else:
            x_hat = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)
        out_data = self.gamma.data * x_hat + self.beta.data
        out = Tensor(out_data, (x, self.gamma, self.beta), 'batchnorm')
        # backward omitted for brevity
        return out
    def parameters(self): return [self.gamma, self.beta]
